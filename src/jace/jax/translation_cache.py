# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains the functionality related to the compilation cache of the stages.

Actually there are two different caches:
- The lowering cache.
- And the compilation cache.

Both are implemented as a singleton.
"""

from __future__ import annotations

import functools as ft
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeAlias, runtime_checkable

import dace
from jax import core as jax_core

from jace import util


if TYPE_CHECKING:
    from jace.jax import stages

# This is the default cache size we are using
_DEF_CACHE_SIZE: Final[int] = 256

# This are the caches that we are using.
_TRANSLATION_CACHES: dict[type[stages.Stage], TranslationCache] = {}


def cached_translation(
    action: Callable,
) -> Callable:
    """Decorator for making the transfer method, i.e. `JaceWrapped.lower()` and `JaceLowered.compile()` cacheable.

    The main issue is that we can not simply cache on the actual arguments we pass to them, but on an abstract
    (or concrete; static arguments + compiling) description on them, and this is what this decorator is for.
    Based on its argument it will generate a key of the call, see `TranslationCache.make_key()` for more.
    Then it will check if the result is known and if needed it will perform the actual call.

    Beside this the function will two two things.
    The first is, that it will set the `_cache` member of `self` to the associated cache.
    Thus an annotated object need to define such a member.

    The second thing it will do is optional, if the call is not cached inside the cache the wrapped function has to be run.
    In that case the wrapper will first check if the object defines the `_call_description` member.
    If this is the case the wrapper will set this object to an abstract description of the call, which is also used as key in the cache.
    After the function return this member is set to `None`.
    """

    @ft.wraps(action)
    def _action_wrapper(
        self: stages.JaceWrapped | stages.JaceLowered,
        *args: Any,
        **kwargs: Any,
    ) -> stages.Stage:
        # Get the abstract description of the call, that is used as key.
        key: CachedCallDescription = self._make_call_decscription(*args, **kwargs)
        if self._cache.has(key):
            return self._cache.get(key)

        # We must actually perform the call
        next_stage: stages.Stage = action(self, *args, **kwargs)
        self._cache.add(key, next_stage)
        return next_stage

    return _action_wrapper


def clear_translation_cache() -> None:
    """Clear all caches associated to translation."""
    _TRANSLATION_CACHES.clear()


def get_cache(
    stage: stages.Stage,
) -> TranslationCache:
    """Returns the cache that is used for `stage`."""
    # The caches are per stage and not per instance basis
    tstage = type(stage)
    if tstage not in _TRANSLATION_CACHES:
        _TRANSLATION_CACHES[tstage] = TranslationCache(size=_DEF_CACHE_SIZE)
    return _TRANSLATION_CACHES[tstage]


@dataclass(init=True, eq=True, frozen=True)
class _AbstarctCallArgument:
    """Class to represent one argument to the call in an abstract way.

    It is used as part of the key in the cache.
    It represents the structure of the argument, i.e. its shape, type and so on, but nots its value.
    To construct it you should use the `from_value()` class function which interfere the characteristics from a value.
    """

    shape: tuple[int, ...]
    dtype: dace.typeclass
    strides: tuple[int, ...] | None
    storage: dace.StorageType

    @classmethod
    def from_value(
        cls,
        val: Any,
    ) -> _AbstarctCallArgument:
        """Construct an `_AbstarctCallArgument` from a value.

        Todo:
            Improve, such that NumPy arrays are on CPU, CuPy on GPU and so on.
            This function also probably fails for scalars.
        """
        if not util.is_fully_addressable(val):
            raise NotImplementedError("Distributed arrays are not addressed yet.")
        if isinstance(val, jax_core.Literal):
            raise TypeError("Jax Literals are not supported as cache keys.")

        if util.is_array(val):
            if util.is_jax_array(val):
                val = val.__array__()  # Passing `copy=False` leads to error in NumPy.
            shape = val.shape
            dtype = util.translate_dtype(val.dtype)
            strides = getattr(val, "strides", None)
            # TODO(phimuell): is `CPU_Heap` always okay? There would also be `CPU_Pinned`.
            storage = (
                dace.StorageType.GPU_Global if util.is_on_device(val) else dace.StorageType.CPU_Heap
            )

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        if util.is_scalar(val):
            shape = ()
            dtype = util.translate_dtype(type(val))
            strides = None
            # Lets pretend that scalars are always on the CPU, which is a fair assumption.
            storage = dace.StorageType.CPU_Heap

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        raise TypeError(f"Can not make 'an abstract description from '{type(val).__name__}'.")


@runtime_checkable
class _ConcreteCallArgument(Protocol):
    """Type for encoding a concrete arguments in the cache."""

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


"""This type is the abstract description of a function call.
It is part of the key used in the cache.
"""
CallArgsDescription: TypeAlias = tuple[
    _AbstarctCallArgument
    | _ConcreteCallArgument
    | tuple[str, _AbstarctCallArgument]
    | tuple[str, _ConcreteCallArgument],
    ...,
]


@dataclass(init=True, eq=True, frozen=True)
class CachedCallDescription:
    """Represents the structure of the entire call in the cache and used as key in the cache.

    This class represents both the `JaceWrapped.lower()` and `JaceLowered.compile()` calls.

    The actual key is composed of two parts, first the "origin of the call".
    For this we just use the address of the stage object we are caching and hope that the
    address is not reused for another stag anytime soon.

    The second part is of the key are a description of the actual arguments, see `CallArgsDescription` type alias.
    There are two ways for describing the arguments:
    - `_AbstarctCallArgument`: Which encode only the structure of the arguments.
        These are essentially the tracer used by Jax.
    - `_ConcreteCallArgument`: Which represents actual values of the call.
        These are either the static arguments or compile options.

    While `JaceWrapped.lower()` uses both, `JaceLowered.compile()` will only use concrete arguments.
    In addition an argument can be positional or a named argument,
    in which case it consists of a `tuple[str, _AbstarctCallArgument  | _ConcreteCallArgument]`.

    Notes:
        The base assumption is that the stages are immutable.

    Todo:
        - pytrees.
        - Turn the references into week references, Jax does this and I am sure there is a reason for it.
    """

    stage_id: int
    fargs: CallArgsDescription


class TranslationCache:
    """The _internal_ cache object.

    It implements a simple LRU cache, for storing the results of the `JaceWrapped.lower()` and `JaceLowered.compile()` calls.
    You should not use this cache directly but instead use the `cached_translation` decorator.

    Notes:
        The most recently used entry is at the end of the `OrderedDict`.
            The reason for this is, because there the new entries are added.
    """

    __slots__ = ["_memory", "_size"]

    _memory: OrderedDict[CachedCallDescription, stages.Stage]
    _size: int

    def __init__(
        self,
        size: int,
    ) -> None:
        """Creates a cache instance of size.

        The cache will have size `size` and use `key` as key function.
        """
        if size <= 0:
            raise ValueError(f"Invalid cache size of '{size}'")
        self._memory: OrderedDict[CachedCallDescription, stages.Stage] = OrderedDict()
        self._size = size

    def has(
        self,
        key: CachedCallDescription,
    ) -> bool:
        """Check if `self` have a record of `key`."""
        return key in self._memory

    def get(
        self,
        key: CachedCallDescription,
    ) -> stages.Stage:
        """Get the next stage associated with `key`.

        Notes:
            It is an error if `key` does not exist.
            This function will mark `key` as most recently used.
        """
        if not self.has(key):
            raise KeyError(f"Key '{key}' is unknown.")
        self._memory.move_to_end(key, last=True)
        return self._memory[key]

    def add(
        self,
        key: CachedCallDescription,
        res: stages.Stage,
    ) -> TranslationCache:
        """Adds `res` under `key` to `self`."""
        if self.has(key):
            # `key` is known, so move it to the end and update the mapped value.
            self._memory.move_to_end(key, last=True)
            self._memory[key] = res

        else:
            # `key` is not known so we have to add it
            while len(self._memory) >= self._size:
                self._evict(None)
            self._memory[key] = res
        return self

    def _evict(
        self,
        key: CachedCallDescription | None,
    ) -> None:
        """Evict `key` from `self`.

        If `key` is `None` the oldest entry is evicted.
        """
        if len(self._memory) == 0:
            return
        if key is None:
            self._memory.popitem(last=False)
        elif self.has(key):
            self._memory.move_to_end(key, last=False)
            self._memory.popitem(last=False)

    def __repr__(self) -> str:
        """Textual representation for debugging."""
        return f"TranslationCache({len(self._memory)} / {self._size} || {', '.join( '[' + repr(k) + ']' for k in self._memory)})"
