# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functionality for `jace.jax.jit()`."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import dace
import jax

from jace import util as jutil
from jace.translator import util as jtrutil


class JitWrapped:
    """Result class of all jited functions.

    It is essentially a wrapper around an already jited, i.e. passed to a Jax primitive function.
    The function is then able to compile it if needed.
    However, the wrapped object is itself again tracable, thus it does not break anything.

    Todo:
        Implement a compile cache (shape, data type, strides, location).
        Turn this into a primitive.
        Handles pytrees.
    """

    def __init__(
        self,
        jax_prim: Any,  # No idea if there is a better type.
    ) -> None:
        """Creates a wrapped jace jitable object of `jax_prim`."""
        assert jax_prim is not None
        self._fun = jax_prim
        self._tran_count = 0

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Compile and run the wrapped function.

        In case `self` is called by Jax during a trace, the call will
        transparently forwarded to the wrapped function.
        This guarantees that `self` itself is traceable.
        """

        if jutil.is_tracing_ongoing(*args, **kwargs):
            return self._forward_trace(*args, **kwargs)
        return self._call_sdfg(*args, **kwargs)

    def _forward_trace(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Is called by `self.__call__` if a trace operation was detected.

        I.e. it will simply forward the call to the wrapped function.
        """
        if len(kwargs) != 0:
            raise RuntimeError("Passed kwargs, which are not allowed in tracing.")
        return self._fun(*args, **kwargs)

    def _call_sdfg(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Compiles and run the wrapped function.

        Notes:
            Currently no caching of the compiled object is done.
        """
        from jace.translator.util import debug as jtrudebug

        memento: jtrutil.JaCeTranslationMemento = self._get_memento(*args, **kwargs)
        return jtrudebug.run_memento(memento, *args)

    def _get_memento(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> jtrutil.JaCeTranslationMemento:
        """This function returns the Memento.

        The function will transform its arguments into `_ArgInfo` versions.
        This is needed since Jax only cares about the information stored inside it.
        The positional only arguments are used to cache the settings important for Jax
        and the kwonly arguments are used to influence the Jaxpr to SDFG translator.

        Notes:
            It is forbidden to permanently modify the returned memento.
                Doing so results in undefined behaviour.
        """
        return self._get_memento_cached(
            *(_ArgInfo.from_value(v) for v in args),
            **kwargs,
        )

    @lru_cache
    def _get_memento_cached(
        self,
        *args: _ArgInfo,
        **kwargs: Any,
    ) -> jtrutil.JaCeTranslationMemento:
        """Generates the SDFG from

        Todo:
            Also make the SDFG compiled and permanent also in the memento
            Implement a better cache that avoids using this strange way to pass values around.

        Notes:
            It is forbidden to permanently modify the returned memento.
                Doing so results in undefined behaviour.
        """
        from jace.translator import JaxprTranslationDriver

        real_args: tuple[Any, ...] = tuple(x._get_val_once() for x in args)
        jaxpr = jax.make_jaxpr(self.__wrapped__)(*real_args)
        driver = JaxprTranslationDriver(**kwargs)
        return driver.translate_jaxpr(jaxpr)

    @property
    def __wrapped__(self) -> Any:
        """Returns the wrapped object."""
        return self._fun

    def __hash__(self) -> int:
        """Hash based on the wrapped function (needed for caching)."""
        return hash(self.__wrapped__)

    def __eq__(self, other: Any) -> bool:
        """Wrapped function based equality testing (needed for caching)."""
        if not isinstance(other, JitWrapped):
            return False
        return self.__wrapped__ == other.__wrapped__


class _ArgInfo:
    """Abstracts argument for the case of the `JitWrapped` object.

    Essentially represents a single argument.
    To construct it use the `from_value()` function.

    Notes:
        An `_ArgInfo` instance also keeps a reference to the value that was used to construct it.
            However this value can only retrieved once and is removed afterwards.
            Conceptionally it should be a weak reerence, but several classes (especially `int`
            and `float` can not be weakly referenced.
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: dace.typeclass
    location: dace.StorageType  # We only need CPU and GPU.
    _val: Any | None  # May not be allocated.

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """To construct an `_ArgInfo` instance use `from_val()`."""
        raise NotImplementedError("Use '_ArgInfo.from_value()' to construct an instance.")

    def __hash__(self) -> int:
        return hash((self.shape, self.strides, self.dtype, self.location))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _ArgInfo):
            return False
        return (self.shape, self.strides, self.dtype, self.location) == (
            (other.shape, other.strides, other.dtype, other.location)
        )

    def _get_val_once(self) -> Any:
        """Returns the wrapped object.

        This function only works for a single time.
        Calling it will null the reference of `self`.
        """
        if self._val is None:
            raise RuntimeError("Value was already consumed.")
        val = self._val
        self._val = None
        return val

    @classmethod
    def from_value(cls, val: Any) -> _ArgInfo:
        """Constructs an `_ArgInfo` instance from `val`."""
        arginfo: _ArgInfo = cls.__new__(cls)
        raise NotImplementedError("'_ArgInfo.from_value()' is not implemented.")
