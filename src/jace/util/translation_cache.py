# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains the functionality related to the compilation cache of the stages.

The cache currently caches the lowering, i.e. the result of `JaCeWrapped.lower()`
and the compilation, i.e. `JaCeLowered.compile()`. The caches are on a per stage
basis and not on a per instant basis. To make a stage cacheable, it must be
derived from `CachingStage` and its transition function must be decoration with
`@cached_transition`.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import functools
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, Any, Concatenate, Generic, ParamSpec, TypeAlias, TypeVar, cast

import dace
from jax import core as jax_core, tree_util as jax_tree

from jace import util


if TYPE_CHECKING:
    from jace import stages

_TRANSLATION_CACHES: dict[type[CachingStage], StageCache] = {}
"""Caches used to store the state transition.

The caches are on a per stage and not per instant basis.
"""


# Type annotation for the caching.
P = ParamSpec("P")
NextStage = TypeVar("NextStage", bound="stages.Stage")
TransitionFunction: TypeAlias = "Callable[Concatenate[CachingStage[NextStage], P], NextStage]"
CachingStageT = TypeVar("CachingStageT", bound="CachingStage")

# Type to describe a single argument either in an abstract or concrete way.
CallArgsSpec: TypeAlias = tuple["_AbstractCallArgument | Hashable"]


class CachingStage(Generic[NextStage]):
    """
    Annotates a stage whose transition to the next stage is cacheable.

    To make a transition cacheable, a stage must:
    - be derived from this class.
    - its `__init__()` function must explicitly call `CachingStage.__init__()`.
    - the transition function must be annotated by `@cached_transition`.
    - it must implement the `_make_call_description()` to create the key.
    - the stage object must be immutable.

    Todo:
        - Handle eviction from the cache due to collecting of unused predecessor stages.
    """

    _cache: StageCache[NextStage]

    def __init__(self) -> None:
        self._cache = get_cache(self)

    @abc.abstractmethod
    def _make_call_description(
        self: CachingStage, in_tree: jax_tree.PyTreeDef, flat_call_args: Sequence[Any]
    ) -> StageTransformationSpec:
        """
        Computes the key used to represent the call.

        This function is used by the `@cached_transition` decorator to perform
        the lookup inside the cache. It should return a description of the call
        that is encapsulated inside a `StageTransformationSpec` object, see
        there for more information.

        Args:
            in_tree: Pytree object describing how the input arguments were flattened.
            flat_call_args: The flattened arguments that were passed to the
                annotated function.
        """
        ...


def cached_transition(
    transition: Callable[Concatenate[CachingStageT, P], NextStage],
) -> Callable[Concatenate[CachingStage[NextStage], P], NextStage]:
    """
    Decorator for making the transition function of the stage cacheable.

    See the description of `CachingStage` for the requirements.
    The function will use `_make_call_description()` to decide if the call is
    already known and if so it will return the cached object. If the call is
    not known it will call the wrapped transition function and record its
    return value inside the cache, before returning it.

    Todo:
        - Implement a way to temporary disable the cache.
    """

    @functools.wraps(transition)
    def transition_wrapper(self: CachingStageT, *args: P.args, **kwargs: P.kwargs) -> NextStage:
        flat_call_args, in_tree = jax_tree.tree_flatten((args, kwargs))
        key = self._make_call_description(flat_call_args=flat_call_args, in_tree=in_tree)
        if key not in self._cache:
            self._cache[key] = transition(self, *args, **kwargs)
        return self._cache[key]

    return cast(TransitionFunction, transition_wrapper)


def clear_translation_cache() -> None:
    """Clear all caches associated to translation."""
    for stage_caches in _TRANSLATION_CACHES.values():
        stage_caches.clear()


def get_cache(stage: CachingStage) -> StageCache:
    """Returns the cache that should be used for `stage`."""
    stage_type = type(stage)
    if stage_type not in _TRANSLATION_CACHES:
        _TRANSLATION_CACHES[stage_type] = StageCache()
    return _TRANSLATION_CACHES[stage_type]


@dataclasses.dataclass(frozen=True)
class _AbstractCallArgument:
    """
    Class to represent a single argument to the transition function in an abstract way.

    As noted in `StageTransformationSpec` there are two ways to describe an
    argument, either by using its concrete value or an abstract description,
    which is similar to tracers in JAX. This class represents the second way.
    To create an instance you should use `_AbstractCallArgument.from_value()`.

    Attributes:
        shape: In case of an array its shape, in case of a scalar the empty tuple.
        dtype: The DaCe type of the argument.
        strides: The strides of the argument, or `None` if they are unknown or a scalar.
        storage: The storage type where the argument is stored.

    Note:
        This class is only able to describe scalars and arrays, thus it should
        only be used after the arguments were flattened.
    """

    shape: tuple[int, ...]
    dtype: dace.typeclass
    strides: tuple[int, ...] | None
    storage: dace.StorageType

    @classmethod
    def from_value(cls, value: Any) -> _AbstractCallArgument:
        """Construct an `_AbstractCallArgument` from `value`."""
        if not util.is_fully_addressable(value):
            raise NotImplementedError("Distributed arrays are not addressed yet.")
        if isinstance(value, jax_core.Literal):
            raise TypeError("JAX Literals are not supported as cache keys.")

        if util.is_array(value):
            if util.is_jax_array(value):
                value = value.__array__()  # Passing `copy=False` leads to error in NumPy.
            shape = value.shape
            dtype = util.translate_dtype(value.dtype)
            strides = util.get_strides_for_dace(value)
            # TODO(phimuell): `CPU_Heap` vs. `CPU_Pinned`.
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


@dataclasses.dataclass(frozen=True)
class StageTransformationSpec:
    """
    Represents the entire call to a state transformation function of a stage.

    State transition functions are annotated with `@cached_transition` and their
    result is cached. They key to locate them inside the cache is represented
    by this class and computed by the `CachingStage._make_call_description()`
    function. The actual key is consists of three parts, `stage_id`, `call_args`
    and `in_tree`, see below for more.

    Args:
        stage_id: Origin of the call, for which the id of the stage object should
            be used.
        flat_call_args: Flat representation of the arguments of the call. Each element
            describes a single argument. To describe an argument there are two ways:
            - Abstract description: In this way, the actual value of the argument
                is irrelevant, its structure is important, similar to the tracers
                used in JAX. To represent it, use `_AbstractCallArgument`.
            - Concrete description: Here the actual value of the argument is
                considered, this is similar to how static arguments in JAX works.
                The only requirement is that they can be hashed.
        in_tree: A pytree structure that describes how the input was flatten.
    """

    stage_id: int
    flat_call_args: CallArgsSpec
    in_tree: jax_tree.PyTreeDef


#: Denotes the stage that is stored inside the cache.
StageT = TypeVar("StageT", bound="stages.Stage")


class StageCache(Generic[StageT]):
    """
    Simple LRU cache to cache the results of the stage transition function.

    Args:
        capacity: The size of the cache, defaults to 256.
    """

    # The most recently used entry is at the end of the `OrderedDict`.
    _memory: collections.OrderedDict[StageTransformationSpec, StageT]
    _capacity: int

    def __init__(
        self,
        capacity: int = 256,
    ) -> None:
        self._capacity = capacity
        self._memory = collections.OrderedDict()

    def __contains__(self, key: StageTransformationSpec) -> bool:
        return key in self._memory

    def __getitem__(self, key: StageTransformationSpec) -> StageT:
        if key not in self:
            raise KeyError(f"Key '{key}' is unknown.")
        self._memory.move_to_end(key, last=True)
        return self._memory[key]

    def __setitem__(self, key: StageTransformationSpec, res: StageT) -> None:
        if key in self:
            self._memory.move_to_end(key, last=True)
            self._memory[key] = res
        else:
            if len(self._memory) == self._capacity:
                self.popitem(None)
            self._memory[key] = res

    def popitem(self, key: StageTransformationSpec | None) -> None:
        """
        Evict `key` from `self`.

        If `key` is `None` the oldest entry is evicted.
        """
        if len(self._memory) == 0:
            return
        if key is None:
            self._memory.popitem(last=False)
        elif key in self:
            self._memory.move_to_end(key, last=False)
            self._memory.popitem(last=False)

    def clear(self) -> None:  # noqa: D102 [undocumented-public-method]
        self._memory.clear()

    def __len__(self) -> int:
        return len(self._memory)

    @property
    def capacity(self) -> int:  # noqa: D102 [undocumented-public-method]
        return self._capacity

    def front(self) -> tuple[StageTransformationSpec, StageT]:
        """Returns the front of the cache, i.e. its newest entry."""
        return next(reversed(self._memory.items()))

    def __repr__(self) -> str:
        return f"StageCache({len(self._memory)} / {self._capacity} || {', '.join('[' + repr(k) + ']' for k in self._memory)})"
