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
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeAlias,
    TypeVar,
    cast,
)

import dace
from jax import core as jax_core

from jace import util


if TYPE_CHECKING:
    from jace.jax import stages

#: Caches used to store the state transition.
#: The states are on a per stage and not per instant basis.
_TRANSLATION_CACHES: dict[type[CachingStage], StageCache] = {}


# Denotes the stage that follows the current one.
#  Used by the `NextStage` Mixin.
NextStage = TypeVar("NextStage", bound="stages.Stage")


class CachingStage(Generic[NextStage]):
    """Annotates a stage whose transition to the next one is cacheable.

    To make a transition function cacheable it must be annotated by the
    `@cached_transition` decorator.
    """

    _cache: StageCache[NextStage]

    def __init__(self) -> None:
        self._cache = get_cache(self)

    @abc.abstractmethod
    def _make_call_description(
        self: CachingStage,
        *args: Any,
        **kwargs: Any,
    ) -> StageTransformationDescription:
        """Generates the key that is used to store/locate the call in the cache."""
        ...


Action_T = TypeVar("Action_T", bound=Callable[..., Any])


def cached_transition(
    action: Action_T,
) -> Action_T:
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
    ):
        key: StageTransformationDescription = self._make_call_description(*args, **kwargs)
        if key in self._cache:
            return self._cache[key]
        next_stage: stages.Stage = action(self, *args, **kwargs)
        self._cache[key] = next_stage
        return next_stage

    return cast(Action_T, _action_wrapper)


def clear_translation_cache() -> None:
    """Clear all caches associated to translation."""
    _TRANSLATION_CACHES.clear()


def get_cache(
    stage: CachingStage,
) -> StageCache:
    """Returns the cache that is used for `stage`."""
    stage_type = type(stage)
    if stage_type not in _TRANSLATION_CACHES:
        _TRANSLATION_CACHES[stage_type] = StageCache()
    return _TRANSLATION_CACHES[stage_type]


@dataclasses.dataclass(frozen=True)
class _AbstractCallArgument:
    """Class to represent one argument to the call in an abstract way.

    It is used as part of the key in the cache.
    It represents the structure of the argument, i.e. its shape, type and so on, but nots its value.
    To construct it you should use the `from_value()` class function which interfere the characteristics from a value.

    Attributes:
        shape:      In case of an array its shape, in case of a scalar the empty tuple.
        dtype:      The DaCe type of the argument.
        strides:    The strides of the argument, or `None` if they are unknown or a scalar.
        storage:    The storage type where the argument is stored.
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
                val = val.__array__()  # Passing `copy=False` leads to error in NumPy.
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
class StageTransformationDescription:
    """Represents the call to a state transformation function.

    State transition functions are annotated with `@cached_transition` and stored inside a cache.
    This class serves as a key inside this cache and is generated by `CachingStage._make_call_description()`.
    The actual key is consists of two parts.

    Attributes:
        stage_id:   Origin of the call, for which the id of the stage object should be used.
        call_args:  Description of the arguments of the call. There are two ways to describe
            the arguments:
            - Abstract description: In this way, the actual value of the argument is irrelevant,
                only the structure of them are important, similar to the tracers used in Jax.
            - Concrete description: Here one caches on the actual value of the argument.
                The only requirement is that they can be hashed.

    Notes:
        The base assumption is that the stages are immutable.

    Todo:
        - pytrees.
    """

    stage_id: int
    call_args: CallArgsDescription


# Denotes the stage that is stored inside the cache.
StageType = TypeVar("StageType", bound="stages.Stage")


class StageCache(Generic[StageType]):
    """LRU cache that is used to cache the stage transitions, i.e. lowering and compiling, in Jace.

    Notes:
        The most recently used entry is at the end of the `OrderedDict`.
    """

    _memory: collections.OrderedDict[StageTransformationDescription, StageType]
    _size: int

    def __init__(
        self,
        size: int = 256,
    ) -> None:
        """Creates a LRU cache with `size` many entries.

        Args:
            size:   Number of entries the cache holds, defaults to 256.
        """
        self._memory = collections.OrderedDict()
        self._size = size

    def __contains__(
        self,
        key: StageTransformationDescription,
    ) -> bool:
        return key in self._memory

    def __getitem__(
        self,
        key: StageTransformationDescription,
    ) -> StageType:
        if key not in self:
            raise KeyError(f"Key '{key}' is unknown.")
        self._memory.move_to_end(key, last=True)
        return self._memory[key]

    def __setitem__(
        self,
        key: StageTransformationDescription,
        res: StageType,
    ) -> None:
        if key in self:
            self._memory.move_to_end(key, last=True)
            self._memory[key] = res
        else:
            if len(self._memory) == self._size:
                self.popitem(None)
            self._memory[key] = res

    def popitem(
        self,
        key: StageTransformationDescription | None,
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
        return f"StageCache({len(self._memory)} / {self._size} || {', '.join( '[' + repr(k) + ']' for k in self._memory)})"
