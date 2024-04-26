# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functionality for `jace.jax.jit()`."""

from __future__ import annotations

from typing import Any

from jace import util as jutil


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

        return jtrudebug._jace_run(self._fun, *args, **kwargs)

    @property
    def __wrapped__(self) -> Any:
        """Returns the wrapped object."""
        return self._fun
