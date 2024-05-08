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
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dace
from jax import core as jax_core

from jace import util
from jace.jax import stages


def get_cache(
    name: str,
    size: int = 128,
) -> TranslationCache:
    """Returns the cache associated to `name`.

    If called for the first time, the cache sizes will be set to `size`.
    In all later calls this value is ignored.
    """
    # Get the caches and if not present, create them.
    if not hasattr(get_cache, "_caches"):
        _caches: dict[str, TranslationCache] = {}
        _caches["lowering"] = TranslationCache(size=size)
        _caches["compiling"] = TranslationCache(size=size)
        get_cache._caches = _caches  # type: ignore[attr-defined]  # ruff removes the `getattr()` calls
    _caches = get_cache._caches  # type: ignore[attr-defined]

    if name not in _caches:
        raise ValueError(f"The cache '{name}' is unknown.")
    return _caches[name]


def cached_translation(
    action: Callable,
) -> Callable:
    """Decorator for making the function cacheable."""

    @ft.wraps(action)
    def _action_wrapper(
        self: stages.Stage,
        *args: Any,
        **kwargs: Any,
    ) -> stages.Stage:
        assert hasattr(self, "_cache")
        cache: TranslationCache = self._cache
        key: _CacheKey = self.make_key(self, *args, **kwargs)
        if cache.has(key):
            return cache.get(key)

        next_stage: stages.Stage = action(*args, **kwargs)
        cache.add(key, next_stage)
        return next_stage

    return _action_wrapper


@util.dataclass_with_default_init(init=True, repr=True, frozen=True, slots=True)
class _JaCeVarWrapper:
    """Wrapper class around `JaCeVar` for use in `_CacheKey`.

    It essentially makes the hash depend on the content, with the exception of name.
    """

    _slots__ = ("var", "_hash")
    var: util.JaCeVar
    _hash: int

    def __init__(self, var: util.JaCeVar) -> None:
        _hash: int = hash((var.shape, var.dtype, var.storage))
        if var.name != "":
            raise ValueError("Key can not have a name.")
        self.__default_init__(var=var, _hash=_hash)  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _JaCeVarWrapper):
            return NotImplemented
        return (self.var.shape, self.var.dtype, self.var.storage) == (
            other.var.shape,
            other.var.dtype,
            other.var.storage,
        )

    @classmethod
    def from_value(
        cls,
        val: Any,
    ) -> _JaCeVarWrapper:
        """Returns a `JaCe` variable constructed from `val`.

        If `val` is on the device, its storage type will be `GPU_Global` otherwise the default.

        Todo:
            Improve, such that NumPy arrays are on CPU, CuPy on GPU and so on.
        """
        if not util.is_fully_addressable(val):
            raise NotImplementedError("Distributed arrays are not addressed yet.")

        if isinstance(val, util.JaCeVar):
            return cls(var=val)

        # Define the storage as given by on device.
        storage: dace.StorageType | None = (
            dace.StorageType.GPU_Global if util.is_on_device(val) else None
        )

        if isinstance(val, jax_core.Var):
            val = val.aval
        if isinstance(val, jax_core.Literal):
            raise TypeError("Jax Literals are not supported as cache keys.")

        # We need at least a shaped array
        if isinstance(val, jax_core.ShpedArray):
            return cls(
                util.JaCeVar(
                    name="",
                    shape=val.aval.shape,
                    dtype=val,
                    storage=storage,
                ),
            )
        if isinstance(val, jax_core.AbstractValue):
            raise TypeError(f"Can not make 'JaCeVar' from '{type(val).__name__}', too abstract.")

        # If we are here, then we where not able, thus we will will now try Jax
        #  This is inefficient and we should make it better.
        return cls.from_value(jax_core.get_aval(val))


@dataclass(init=True, eq=True, frozen=True, unsafe_hash=True)
class _CacheKey:
    """Wrapper around the arguments"""

    __slots__ = ("fun", "sdfg_hash", "vars", "_hash")

    # Note that either `_fun` or `_sdfg_hash` are not `None`.
    fun: Callable | None
    sdfg_hash: int | None
    fargs: tuple[_JaCeVarWrapper, ...]

    @classmethod
    def Create(
        cls,
        stage: stages.Stage,
        *args: Any,
        **kwargs: Any,
    ) -> _CacheKey:
        """Creates a cache key for the stage object `stage` that was called to advance."""
        if len(kwargs) != 0:
            raise NotImplementedError("kwargs are not implemented.")

        if isinstance(stage, stages.JaceWrapped):
            fun = stage.__wrapped__
            sdfg_hash = None
        elif isinstance(stage, stages.JaceLowered):
            fun = None
            sdfg_hash = int(stage.compiler_ir().sdfg.hash_sdfg, 16)
        else:
            raise TypeError(f"Can not make key from '{type(stage).__name__}'.")

        fargs = tuple(_JaCeVarWrapper.from_value(x) for x in args)

        return cls(fun=fun, sdfg_hash=sdfg_hash, fargs=fargs)


class TranslationCache:
    """The _internal_ cache object.

    It implements a simple LRU cache.

    Todo:
        Also handle abstract values.
    """

    __slots__ = ["_memory", "_size"]

    _memory: OrderedDict[_CacheKey, stages.Stage]
    _size: int

    def __init__(
        self,
        size: int = 128,
    ) -> None:
        """Creates a cache instance of size `size`."""
        if size <= 0:
            raise ValueError(f"Invalid cache size of '{size}'")
        self._memory: OrderedDict[_CacheKey, stages.Stage] = OrderedDict()
        self._size = size

    @staticmethod
    def make_key(
        stage: stages.Stage,
        *args: Any,
        **kwargs: Any,
    ) -> _CacheKey:
        """Create a key object for `stage`."""
        if len(kwargs) != 0:
            raise NotImplementedError
        return _CacheKey.Create(stage, *args, **kwargs)

    def has(
        self,
        key: _CacheKey,
    ) -> bool:
        """Check if `self` have a record of `key`.

        To generate `key` use the `make_key` function.
        """
        return key in self._memory

    def get(
        self,
        key: _CacheKey,
    ) -> stages.Stage:
        """Get the next stage associated with `key`.

        It is an error if `key` does not exists.
        This function will move `key` to front of `self`.
        """
        if not self.has(key):
            raise KeyError(f"Key '{key}' is unknown.")
        self._memory.move_to_end(key, last=False)
        return self._memory.get(key)  # type: ignore[return-value]  # type confusion

    def add(
        self,
        key: _CacheKey,
        res: stages.Stage,
    ) -> TranslationCache:
        """Adds `res` under `key` to `self`.

        In case `key` is already known, it will first be eviceted and then reinserted.
        If `self` is larger than specified the oldest one will be evicted.
        """
        self._evict(key)
        while len(self._memory) >= self._size:
            self._memory.popitem(last=True)
        self._memory[key] = res
        self._memory.move_to_end(key, last=False)
        return self

    def _evict(
        self,
        key: _CacheKey,
    ) -> bool:
        """Evict `key` from `self`.

        Returns if it was evicted or not.
        """
        if not self.has(key):
            return False
        self._memory.move_to_end(key, last=True)
        self._memory.popitem(last=True)
        return True
