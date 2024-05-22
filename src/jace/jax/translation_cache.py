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

import abc
import collections
import dataclasses
import functools
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import dace
from jax import core as jax_core

from jace import util


if TYPE_CHECKING:
    from jace.jax import stages

# This is the default cache size we are using
_DEF_CACHE_SIZE: Final[int] = 256

# This are the caches that we are using.
_TRANSLATION_CACHES: dict[type[CachingStage], TranslationCache] = {}


class CachingStage:
    """Annotates a stage whose transition to the next one is cacheable.

    This transitions are mainly `JaceWrapped.lower()` and `JaceLowered.compile()` calls.
    To make a stage cacheable annotate the transition function with the `@cached_translation` decorator.

    Todo:
        - Make a generic to indicate what the result stage is.
    """

    _cache: TranslationCache

    def __init__(self) -> None:
        self._cache = get_cache(self)

    @abc.abstractmethod
    def _make_call_description(
        self: CachingStage,
        *args: Any,
        **kwargs: Any,
    ) -> CachedCallDescription:
        """Generates the key that is used to store/locate the call in the cache."""
        ...


def cached_translation(
    action: Callable[..., stages.Stage],
) -> Callable:
    """Decorator for making the transition function of the stage cacheable.

    The decorator will call the annotated function only if the call is not stored inside the cache.
    The key to look up the call in the cache is computed by `self._make_call_description()`.
    For this the stage must be derived from `CachingStage`.
    """

    @functools.wraps(action)
    def _action_wrapper(
        self: CachingStage,
        *args: Any,
        **kwargs: Any,
    ) -> stages.Stage:
        # Get the abstract description of the call, that is used as key.
        key: CachedCallDescription = self._make_call_description(*args, **kwargs)
        if key in self._cache:
            return self._cache[key]

        # We must actually perform the call
        next_stage: stages.Stage = action(self, *args, **kwargs)
        self._cache[key] = next_stage
        return next_stage

    return _action_wrapper


def clear_translation_cache() -> None:
    """Clear all caches associated to translation."""
    _TRANSLATION_CACHES.clear()


def get_cache(
    stage: CachingStage,
) -> TranslationCache:
    """Returns the cache that is used for `stage`."""
    # The caches are per stage and not per instance basis
    tstage = type(stage)
    if tstage not in _TRANSLATION_CACHES:
        _TRANSLATION_CACHES[tstage] = TranslationCache(size=_DEF_CACHE_SIZE)
    return _TRANSLATION_CACHES[tstage]


@dataclasses.dataclass(frozen=True)
class _AbstractCallArgument:
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
    ) -> _AbstractCallArgument:
        """Construct an `_AbstractCallArgument` from a value.

        Todo:
            Handle storage location of arrays correctly.
        """
        if not util.is_fully_addressable(val):
            raise NotImplementedError("Distributed arrays are not addressed yet.")
        if isinstance(val, jax_core.Literal):
            raise TypeError("Jax Literals are not supported as cache keys.")

        if util.is_array(val):
            if util.is_jax_array(val):
                val = val.__array__(copy=False)
            shape = val.shape
            dtype = util.translate_dtype(val.dtype)
            strides = getattr(val, "strides", None)
            # Is `CPU_Heap` always okay? There would also be `CPU_Pinned`.
            storage = (
                dace.StorageType.GPU_Global if util.is_on_device(val) else dace.StorageType.CPU_Heap
            )

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        if util.is_scalar(val):
            shape = ()
            dtype = util.translate_dtype(type(val))
            strides = None
            # Scalar arguments are always on the CPU and never on the GPU.
            storage = dace.StorageType.CPU_Heap

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        raise TypeError(f"Can not make 'an abstract description from '{type(val).__name__}'.")


#: This type is the abstract description of a function call.
#:  It is part of the key used in the cache.
CallArgsDescription: TypeAlias = tuple[
    _AbstractCallArgument | Hashable | tuple[str, _AbstractCallArgument | Hashable],
    ...,
]


@dataclasses.dataclass(frozen=True)
class CachedCallDescription:
    """Represents the full structure of a call in the cache as a key.

    This class is the return type of the `CachingStage._make_call_description()` function,
    which is used by the `@cached_translation` decorator to compute a key of transition.
    This allows to either retrieve or then store the result of the actual call in the cache.

    The actual key is composed of two parts, first the "origin of the call".
    For this we just use the address of the stage object we are caching and hope that the
    address is not reused for another stag anytime soon.

    The second part is of the key are a description of the actual arguments, see `CallArgsDescription` type alias.
    For this the `_make_call_description()` method of the stage is used.
    The arguments can be described in two different ways:
    - Abstract description: In this way, the actual value of the argument is irrelevant,
        only the structure of them are important, this is similar to the tracers used in Jax.
    - Concrete description: Here one caches on the actual value of the argument,
        which is similar to static arguments in Jax.
        The only restriction is that they are hash able.

    Notes:
        The base assumption is that the stages are immutable.

    Todo:
        - pytrees.
        - Turn the references into week references, Jax does this and I am sure there is a reason for it.
    """

    stage_id: int
    fargs: CallArgsDescription


class TranslationCache:
    """The cache object used to cache the stage transitions.

    Notes:
        The most recently used entry is at the end of the `OrderedDict`, because it puts new entries there.
    """

    __slots__ = ("_memory", "_size")

    _memory: collections.OrderedDict[CachedCallDescription, stages.Stage]
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
        self._memory = collections.OrderedDict()
        self._size = size

    def __contains__(
        self,
        key: CachedCallDescription,
    ) -> bool:
        """Check if `self` have a record of `key`."""
        return key in self._memory

    def __getitem__(
        self,
        key: CachedCallDescription,
    ) -> stages.Stage:
        """Get the next stage associated with `key`.

        Notes:
            It is an error if `key` does not exist.
            This function will mark `key` as most recently used.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' is unknown.")
        self._memory.move_to_end(key, last=True)
        return self._memory[key]

    def __setitem__(
        self,
        key: CachedCallDescription,
        res: stages.Stage,
    ) -> TranslationCache:
        """Adds or update `key` to map to `res`."""
        if key in self:
            # `key` is known, so move it to the end and update the mapped value.
            self._memory.move_to_end(key, last=True)
            self._memory[key] = res

        else:
            # `key` is not known so we have to add it
            while len(self._memory) >= self._size:
                self.popitem(None)
            self._memory[key] = res
        return self

    def popitem(
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
        elif key in self:
            self._memory.move_to_end(key, last=False)
            self._memory.popitem(last=False)

    def __repr__(self) -> str:
        """Textual representation for debugging."""
        return f"TranslationCache({len(self._memory)} / {self._size} || {', '.join( '[' + repr(k) + ']' for k in self._memory)})"
