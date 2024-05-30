# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains the functionality related to the compilation cache of the stages.

The cache currently caches the lowering, i.e. the result of `JaCeWrapped.lower()` and the
compilation, i.e. `JaCeLowered.compile()`. The caches are on a per stage basis and not on a
per instant basis. To make a stage cacheable, it must be derived from `CachingStage` and
its transition function must be decoration with `@cached_transition`.
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
    Concatenate,
    Generic,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
)

import dace
from jax import core as jax_core

from jace import util


if TYPE_CHECKING:
    from jace import stages

#: Caches used to store the state transition.
#: The caches are on a per stage and not per instant basis.
_TRANSLATION_CACHES: dict[type[CachingStage], StageCache] = {}


# Denotes the stage that follows the current one.
#  Used by the `NextStage` Mixin.
NextStage = TypeVar("NextStage", bound="stages.Stage")


class CachingStage(Generic[NextStage]):
    """Annotates a stage whose transition to the next stage is cacheable.

    To make the transition of a stage cacheable, the stage must be derived from this class,
    and its initialization must call `CachingStage.__init__()`. Furthermore, its transition
    function must be annotated by the `@cached_transition` decorator.

    A class must implement the `_make_call_description()` to compute an abstract description
    of the call. This is needed to operate the cache to store the stage transitions.

    Notes:
        The `__init__()` function must explicitly be called to fully setup `self`.
    """

    _cache: StageCache[NextStage]

    def __init__(self) -> None:
        self._cache = get_cache(self)

    @abc.abstractmethod
    def _make_call_description(
        self: CachingStage,
        *args: Any,
        **kwargs: Any,
    ) -> StageTransformationSpec:
        """Generates the key that is used to store/locate the call in the cache."""
        ...


# Type annotation of the caching Stuff.
P = ParamSpec("P")
TransitionFunction = Callable[Concatenate[CachingStage[NextStage], P], NextStage]
CachingStageType = TypeVar("CachingStageType", bound=CachingStage)


def cached_transition(
    transition: Callable[Concatenate[CachingStageType, P], NextStage],
) -> Callable[Concatenate[CachingStage[NextStage], P], NextStage]:
    """Decorator for making the transition function of the stage cacheable.

    In order to work, the stage must be derived from `CachingStage`. For computing the key of a
    call the function will use the `_make_call_description()` function of the cache.
    """

    @functools.wraps(transition)
    def transition_wrapper(
        self: CachingStageType,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> NextStage:
        key: StageTransformationSpec = self._make_call_description(*args, **kwargs)
        if key in self._cache:
            return self._cache[key]
        next_stage = transition(self, *args, **kwargs)
        self._cache[key] = next_stage
        return next_stage

    return cast(TransitionFunction, transition_wrapper)


def clear_translation_cache() -> None:
    """Clear all caches associated to translation."""
    for stage_caches in _TRANSLATION_CACHES.values():
        stage_caches.clear()


def get_cache(
    stage: CachingStage,
) -> StageCache:
    """Returns the cache that should be used for `stage`."""
    stage_type = type(stage)
    if stage_type not in _TRANSLATION_CACHES:
        _TRANSLATION_CACHES[stage_type] = StageCache()
    return _TRANSLATION_CACHES[stage_type]


@dataclasses.dataclass(frozen=True)
class _AbstractCallArgument:
    """Class to represent a single argument to the transition function in an abstract way.

    As noted in `StageTransformationSpec` there are two ways to describe an argument,
    either using its concrete value or an abstract description, which is similar to tracers in Jax.
    This class represents the second way.
    To create an instance you should use `_AbstractCallArgument.from_value()`.

    Its description is limited to scalars and arrays. To describe more complex types, they
    should be processed by pytrees first.

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
        value: Any,
    ) -> _AbstractCallArgument:
        """Construct an `_AbstractCallArgument` from `value`."""
        if not util.is_fully_addressable(value):
            raise NotImplementedError("Distributed arrays are not addressed yet.")
        if isinstance(value, jax_core.Literal):
            raise TypeError("Jax Literals are not supported as cache keys.")

        if util.is_array(value):
            if util.is_jax_array(value):
                value = value.__array__()  # Passing `copy=False` leads to error in NumPy.
            shape = value.shape
            dtype = util.translate_dtype(value.dtype)
            strides = getattr(value, "strides", None)
            # Is `CPU_Heap` always okay? There would also be `CPU_Pinned`.
            storage = (
                dace.StorageType.GPU_Global
                if util.is_on_device(value)
                else dace.StorageType.CPU_Heap
            )

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        if util.is_scalar(value):
            shape = ()
            dtype = util.translate_dtype(type(value))
            strides = None
            # Scalar arguments are always on the CPU and never on the GPU.
            storage = dace.StorageType.CPU_Heap

            return cls(shape=shape, dtype=dtype, strides=strides, storage=storage)

        raise TypeError(f"Can not make 'an abstract description from '{type(value).__name__}'.")


#: This type is the abstract description of a function call.
#:  It is part of the key used in the cache.
CallArgsSpec: TypeAlias = tuple[
    _AbstractCallArgument | Hashable | tuple[str, _AbstractCallArgument | Hashable],
    ...,
]


@dataclasses.dataclass(frozen=True)
class StageTransformationSpec:
    """Represents the entire call to a state transformation function of a stage.

    State transition functions are annotated with `@cached_transition` and their result may be
    cached. They key to locate them inside the cache is represented by this class.
    The cache will call the `CachingStage._make_call_description()` function to get a key.
    The actual key is consists of two parts, `stage_id` and `call_args`.

    Args:
        stage_id:   Origin of the call, for which the id of the stage object should be used.
        call_args:  Description of the arguments of the call. There are two ways to describe
            the arguments:
            - Abstract description: In this way, the actual value of the argument is irrelevant,
                only the structure of them are important, similar to the tracers used in Jax.
            - Concrete description: Here one caches on the actual value of the argument.
                The only requirement is that they can be hashed.

    Todo:
        In the future pytrees will be used as third part.
    """

    stage_id: int
    call_args: CallArgsSpec


# Denotes the stage that is stored inside the cache.
StageType = TypeVar("StageType", bound="stages.Stage")


class StageCache(Generic[StageType]):
    """Simple LRU cache to cache the results of the stage transition function.

    Args:
        size:   The size of the cache, defaults to 256.

    Notes:
        The most recently used entry is at the end of the `OrderedDict`.
    """

    _memory: collections.OrderedDict[StageTransformationSpec, StageType]
    _size: int

    def __init__(
        self,
        size: int = 256,
    ) -> None:
        self._memory = collections.OrderedDict()
        self._size = size

    def __contains__(
        self,
        key: StageTransformationSpec,
    ) -> bool:
        return key in self._memory

    def __getitem__(
        self,
        key: StageTransformationSpec,
    ) -> StageType:
        if key not in self:
            raise KeyError(f"Key '{key}' is unknown.")
        self._memory.move_to_end(key, last=True)
        return self._memory[key]

    def __setitem__(
        self,
        key: StageTransformationSpec,
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
        key: StageTransformationSpec | None,
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

    def clear(self) -> None:
        self._memory.clear()

    def __repr__(self) -> str:
        return f"StageCache({len(self._memory)} / {self._size} || {', '.join( '[' + repr(k) + ']' for k in self._memory)})"
