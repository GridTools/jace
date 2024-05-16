# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the implementation of the jit functioanlity of JaCe."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax as _jax_jax

from jace import jax as jjax, translator
from jace.jax import api_helper


@api_helper.jax_wrapper(_jax_jax.jit)
def jit(
    fun: Callable | None = None,
    /,
    sub_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> jjax.JaceWrapped:
    """Jace's replacement for `jax.jit` (just-in-time) wrapper.

    It works the same way as `jax.jit` does, but instead of using XLA the computation is lowered to DaCe.
    It supports the same arguments as `jax.jit` (although currently not) does.
    In addition it accepts some Jace specific arguments.

    Args:
        sub_translators:    Use these subtranslators for the lowering to DaCe.

    Notes:
        If no subtranslators are specified then the ones that are currently active,
            i.e. the output of `get_subtranslators()`, are used.
            After construction the set of subtranslators that are used by the wrapped object can not be changed.
    """
    if any(kwargs.get(arg, None) is not None for arg in ["donate_argnums", "donate_argnames"]):
        # Donated arguments are not yet fully supported, the prototype supported something similar.
        #  However, the documentation mentioned that they are only a hint, thus we ignore them.
        kwargs.pop("donate_argnums", None)
        kwargs.pop("donate_argnames", None)

    if len(kwargs) != 0:
        raise NotImplementedError(
            f"The following arguments of 'jax.jit' are not yet supported by jace: {', '.join(kwargs.keys())}."
        )

    # fmt: off
    if fun is None:
        # TODO: Is there an obscure case where it makes sense to copy `sub_translators`?
        def wrapper(f: Callable) -> jjax.JaceWrapped:
            return jit(f, sub_translators=sub_translators, **kwargs)
        return wrapper  # type: ignore[return-value]
    # fmt: on

    # If no subtranslators were specified then use the ones that are currently installed.
    if sub_translators is None:
        sub_translators = translator.get_subtranslators()

    return jjax.JaceWrapped(
        fun=fun,
        sub_translators=sub_translators,
        jit_ops=kwargs,
    )


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
    return _jax_jax.vmap(
        fun,
        **kwargs,
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
