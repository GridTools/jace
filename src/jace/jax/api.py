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
) -> jjax.JaceWrapped:
    """Creates a jit wrapper instance."""
    import jax

    if any(
        kwargs.get(static, None) is not None for static in ["static_argnums", "static_argnames"]
    ):
        raise NotImplementedError("Static arguments are not yet supported.")

    if fun is None:
        assert len(kwargs) > 0

        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return jit(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    # in case we are dealing with a JaCe object, we first unwrap it.
    #  Recursion to handle arbitrary deep nestings.
    if util.is_jaceified(fun):
        fun = cast(jjax.JaceWrapped, fun)
        return jit(fun.__wrapped__)

    # Prevents the creation of a level of unnecessary jit.
    #  Probably better solution by using the `disable_jit()`?
    if len(kwargs) == 0:
        return jjax.JaceWrapped(fun)
    return jjax.JaceWrapped(jax.jit(fun, **kwargs))


def grad(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """The gradient transformation."""
    import jax

    if fun is None:

        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return grad(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    return jjax.JaceWrapped(jax.grad(fun, **kwargs))


def jacfwd(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Returns the Jacobian of `fun` in forward differentiation mode."""
    import jax

    if fun is None:

        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return jacfwd(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    return jjax.JaceWrapped(jax.jacfwd(fun, **kwargs))


def jacrev(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Returns the Jacobian of `fun` in reverse differentiation mode."""
    import jax

    if fun is None:

        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return jacrev(f, **kwargs)

        return wrapper  # type: ignore[return-value]

    return jjax.JaceWrapped(jax.jacrev(fun, **kwargs))
