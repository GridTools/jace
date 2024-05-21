# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the implementation of the jit functioanlity of JaCe."""

from __future__ import annotations

import functools as ft
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

import jax as _jax_jax

from jace import translator


if TYPE_CHECKING:
    from jace.jax import stages


@overload
def jit(
    fun: Literal[None] = None,
    /,
    sub_translators: Mapping[str, translator.PrimitiveTranslatorCallable] | None = None,
    **kwargs: Any,
) -> Callable[..., stages.JaceWrapped]: ...


@overload
def jit(
    fun: Callable,
    /,
    sub_translators: Mapping[str, translator.PrimitiveTranslatorCallable] | None = None,
    **kwargs: Any,
) -> stages.JaceWrapped: ...


def jit(
    fun: Callable | None = None,
    /,
    sub_translators: Mapping[str, translator.PrimitiveTranslatorCallable] | None = None,
    **kwargs: Any,
) -> stages.JaceWrapped | Callable[..., stages.JaceWrapped]:
    """Jace's replacement for `jax.jit` (just-in-time) wrapper.

    It works the same way as `jax.jit` does, but instead of using XLA the computation is lowered to DaCe.
    It supports the same arguments as `jax.jit` (although currently not) does.
    In addition it accepts some Jace specific arguments.

    Args:
        sub_translators:    Use these subtranslators for the lowering to DaCe.

    Notes:
        If no subtranslators are specified then the ones that are currently active,
            i.e. the output of `get_regsitered_primitive_translators()`, are used.
            After construction changes to the passed `sub_translators` have no effect on the returned object.
    """
    if len(kwargs) != 0:
        raise NotImplementedError(
            f"The following arguments of 'jax.jit' are not yet supported by jace: {', '.join(kwargs.keys())}."
        )

    def wrapper(f: Callable) -> stages.JaceWrapped:
        from jace import jax as stages  # Cyclic import

        jace_wrapper = stages.JaceWrapped(
            fun=f,
            sub_translators=(
                translator.managing._PRIMITIVE_TRANSLATORS_DICT
                if sub_translators is None
                else sub_translators
            ),
            jit_ops=kwargs,
        )
        return ft.wraps(f)(jace_wrapper)

    return wrapper if fun is None else wrapper(fun)


def vmap(
    fun: Callable,
    /,
    **kwargs: Any,
) -> stages.JaceWrapped:
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


def jacfwd(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> Callable:
    """Jace wrapper around `jax.jacfwd`."""
    return _jax_jax.jacfwd(fun, **kwargs)


def jacrev(
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> Callable:
    """Jace wrapper around `jax.jacrev`."""
    return _jax_jax.jacrev(fun, **kwargs)
