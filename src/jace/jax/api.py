# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the implementation of the jit functioanlity of JaCe."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jace import jax as jjax, util


def jit(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Creates a jit wrapper instance."""
    import jax
    from jax._src import sharding_impls

    if any(kwargs.get(arg, None) is not None for arg in ["static_argnums", "static_argnames"]):
        raise NotImplementedError("Static arguments are not yet supported.")
    if any(kwargs.get(arg, None) is not None for arg in ["donate_argnums", "donate_argnames"]):
        # Donated arguments are not yet (fully) supported, since they are more like a "hint"
        #  to jax we will silently ignore them.
        kwargs["donate_argnums"] = None
        kwargs["donate_argnames"] = None
    if any(
        kwargs.get(x, sharding_impls.UNSPECIFIED) is not sharding_impls.UNSPECIFIED
        for x in ["in_shardings", "out_shardings"]
    ):
        raise NotImplementedError("Sharding is not yet supported.")
    if kwargs.get("device", None) is not None:
        raise NotImplementedError("Selecting of device is not yet supported.")
    if kwargs.get("backend", None) is not None:
        raise NotImplementedError("Selecting of backend is not yet supported.")

    # fmt: off
    if fun is None:
        assert len(kwargs) > 0
        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return jit(f, **kwargs)
        return wrapper  # type: ignore[return-value]
    # fmt: on

    if util.is_jaceified(fun):
        return jit(fun.__wrapped__, **kwargs)
    if len(kwargs) == 0:
        # Prevents the creation of a level of unnecessary jit.
        #  TODO(philmuell): Find a better way, probably better hijacking or `inline`.
        return jjax.JaceWrapped(fun)
    return jjax.JaceWrapped(jax.jit(fun, **kwargs))


def grad(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """The gradient transformation.

    Todo:
        Handle controlflow properly (https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-autodiff)
    """
    import jax

    # fmt: off
    if fun is None:
        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return grad(f, **kwargs)
        return wrapper  # type: ignore[return-value]
    # fmt: on

    return jjax.JaceWrapped(jax.grad(fun, **kwargs))


def jacfwd(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Returns the Jacobian of `fun` in forward differentiation mode."""
    import jax

    # fmt: off
    if fun is None:
        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return jacfwd(f, **kwargs)
        return wrapper  # type: ignore[return-value]
    # fmt: on

    return jjax.JaceWrapped(jax.jacfwd(fun, **kwargs))


def jacrev(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Returns the Jacobian of `fun` in reverse differentiation mode."""
    import jax

    # fmt: off
    if fun is None:
        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return jacrev(f, **kwargs)
        return wrapper  # type: ignore[return-value]
    # fmt: on

    return jjax.JaceWrapped(jax.jacrev(fun, **kwargs))
