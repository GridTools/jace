# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jax.*` namespace."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal, overload

from jax import grad, jacfwd, jacrev

from jace import stages, translator


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


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
) -> Callable[[Callable], stages.JaCeWrapped]: ...


@overload
def jit(
    fun: Callable,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> stages.JaCeWrapped: ...


def jit(
    fun: Callable | None = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> stages.JaCeWrapped | Callable[[Callable], stages.JaCeWrapped]:
    """JaCe's replacement for `jax.jit` (just-in-time) wrapper.

    It works the same way as `jax.jit` does, but instead of using XLA the
    computation is lowered to DaCe. In addition it accepts some JaCe specific
    arguments.

    Args:
        primitive_translators: Use these primitive translators for the lowering
            to SDFG. If not specified the translators in the global registry are
            used.

    Notes:
        After constructions any change to `primitive_translators` has no effect.
    """
    if kwargs:
        # TODO(phimuell): Add proper name verification and exception type.
        raise NotImplementedError(
            f"The following arguments to 'jace.jit' are not yet supported: {', '.join(kwargs)}."
        )

    def wrapper(f: Callable) -> stages.JaCeWrapped:
        # TODO: Improve typing, such that signature is attached to the `JaCeWrapped`.
        jace_wrapper = stages.JaCeWrapped(
            fun=f,
            primitive_translators=(
                translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY
                if primitive_translators is None
                else primitive_translators
            ),
            jit_options=kwargs,
        )
        functools.update_wrapper(jace_wrapper, f)
        return jace_wrapper

    return wrapper if fun is None else wrapper(fun)
