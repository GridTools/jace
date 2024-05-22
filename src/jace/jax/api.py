# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the implementation of the jit functioanlity of JaCe."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any, Literal, overload

from jax import grad, jacfwd, jacrev

from jace import translator
from jace.jax import stages


@overload
def jit(
    fun: Literal[None] = None,
    /,
    sub_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> Callable[[Callable], stages.JaceWrapped]: ...


@overload
def jit(
    fun: Callable,
    /,
    sub_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> stages.JaceWrapped: ...


def jit(
    fun: Callable | None = None,
    /,
    sub_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> stages.JaceWrapped | Callable[[Callable], stages.JaceWrapped]:
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
    if kwargs:
        raise NotImplementedError(
            f"The following arguments of 'jax.jit' are not yet supported by jace: {', '.join(kwargs.keys())}."
        )

    def wrapper(f: Callable) -> stages.JaceWrapped:
        jace_wrapper = stages.JaceWrapped(
            fun=f,
            sub_translators=(
                translator.managing._PRIMITIVE_TRANSLATORS_DICT
                if sub_translators is None
                else sub_translators
            ),
            jit_ops=kwargs,
        )
        return functools.update_wrapper(jace_wrapper, f)

    return wrapper if fun is None else wrapper(fun)


__all__ = [
    "grad",
    "jit",
    "jacfwd",
    "jacrev",
]
