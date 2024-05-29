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
from typing import Any, Literal, cast, overload

from jax import grad, jacfwd, jacrev

from jace import stages, translator


__all__ = [
    "grad",
    "jacfwd",
    "jacrev",
    "jit",
]


@overload
def jit(
    fun: Literal[None] = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> Callable[[Callable], stages.JaceWrapped]: ...


@overload
def jit(
    fun: Callable,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> stages.JaceWrapped: ...


def jit(
    fun: Callable | None = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> stages.JaceWrapped | Callable[[Callable], stages.JaceWrapped]:
    """Jace's replacement for `jax.jit` (just-in-time) wrapper.

    It works the same way as `jax.jit` does, but instead of using XLA the computation is lowered
    to DaCe. It supports the same arguments as `jax.jit` (although currently not) does.
    In addition it accepts some Jace specific arguments.

    Args:
        primitive_translators:    Use these primitive translators for the lowering to SDFG.

    Notes:
        If no translators are specified, the ones in the global registry are implicitly passed
        as argument. After constructions any change to `primitive_translators` has no effect.
    """
    if kwargs:
        # TODO(phimuell): Add proper name verification and exception type.
        raise NotImplementedError(
            f"The following arguments to 'jace.jit' are not yet supported: {', '.join(kwargs)}."
        )

    def wrapper(f: Callable) -> stages.JaceWrapped:
        jace_wrapper = stages.JaceWrapped(
            fun=f,
            primitive_translators=(
                translator.managing._PRIMITIVE_TRANSLATORS_DICT
                if primitive_translators is None
                else primitive_translators
            ),
            jit_options=kwargs,
        )
        return cast(stages.JaceWrapped, functools.update_wrapper(jace_wrapper, f))

    return wrapper if fun is None else wrapper(fun)
