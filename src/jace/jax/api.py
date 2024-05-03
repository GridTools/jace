# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the implementation of the jit functioanlity of JaCe."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from jace import jax as jjax, util


def jit(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JitWrapped:
    """Creates a jit wrapper instance."""
    import jax

    if fun is None:
        assert len(kwargs) > 0

        def wrapper(f: Callable) -> jjax.JitWrapped:
            return jit(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    # in case we are dealing with a JaCe object, we first unwrap it.
    #  Recursion to handle arbitrary deep nestings.
    if util.is_jaceified(fun):
        fun = cast(jjax.JitWrapped, fun)
        return jit(fun.__wrapped__)

    # Prevents the creation of a level of unnecessary jit.
    #  Probably better solution by using the `disable_jit()`?
    if len(kwargs) == 0:
        return jjax.JitWrapped(fun)
    return jjax.JitWrapped(jax.jit(fun, **kwargs))


def grad(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JitWrapped:
    """The gradient transformation."""
    import jax

    if fun is None:

        def wrapper(f: Callable) -> jjax.JitWrapped:
            return grad(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    return jjax.JitWrapped(jax.grad(fun, **kwargs))


def jacfwd(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JitWrapped:
    """Returns the Jacobian of `fun` in forward differentiation mode."""
    import jax

    if fun is None:

        def wrapper(f: Callable) -> jjax.JitWrapped:
            return jacfwd(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    return jjax.JitWrapped(jax.jacfwd(fun, **kwargs))


def jacrev(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JitWrapped:
    """Returns the Jacobian of `fun` in reverse differentiation mode."""
    import jax

    if fun is None:

        def wrapper(f: Callable) -> jjax.JitWrapped:
            return jacrev(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    return jjax.JitWrapped(jax.jacrev(fun, **kwargs))
