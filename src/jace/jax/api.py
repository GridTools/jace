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

import jax as _jax_jax

from jace import jax as jjax, util
from jace.jax import api_helper


@api_helper.jax_wrapper(_jax_jax.jit)
def jit(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Jace wrapper for `jax.jit`.

    Wraps the computation `fun` into a wrapped instance, that can either be traced or compiled.
    For more information see `jace.jax.stages`.

    Notes:
        The function can either be used as decorator or as a command.
    """
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


@api_helper.jax_wrapper(_jax_jax.pmap)
def pmap(
    fun: Callable | None = None,  # noqa: ARG001  # Unused argument
    /,
    **kwargs: Any,  # noqa: ARG001 # Unused argument.
) -> jjax.JaceWrapped:
    """Jace wrapper around `jax.pmap`.

    Notes:
        Will be supported in a very late state.
    """
    raise NotImplementedError("Currently Jace is not able to run in multi resource mode.")


@api_helper.jax_wrapper(_jax_jax.vmap)
def vmap(
    fun: Callable,
    /,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Jace wrapper around `jax.vmap`.

    Notes:
        Currently that is an untested extension.
    """
    import warnings

    warnings.warn(
        "You are using the highly untested 'vamp' interface.",
        stacklevel=2,
    )
    return jit(
        _jax_jax.vmap(
            fun,
            **kwargs,
        ),
    )


@api_helper.jax_wrapper(_jax_jax.grad)
def grad(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> Callable:
    """Jace wrapper for `jax.grad`.

    Notes:
        Note we can not put it into a `JaceWrapped` object because in autodiff mode
            control primitives, such as `if` are allowed, but not in `jit`.
            Thus there need to be this extra layer.
    """
    return _jax_jax.grad(fun, **kwargs)


@api_helper.jax_wrapper(_jax_jax.jacfwd)
def jacfwd(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> Callable:
    """Jace wrapper around `jax.jacfwd`."""
    return _jax_jax.jacfwd(fun, **kwargs)


@api_helper.jax_wrapper(_jax_jax.jacrev)
def jacrev(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> Callable:
    """Jace wrapper around `jax.jacrev`."""
    return _jax_jax.jacrev(fun, **kwargs)
