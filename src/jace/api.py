# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jax.*` namespace."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, overload

from jax import grad, jacfwd, jacrev

from jace import stages, translator


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


__all__ = ["grad", "jacfwd", "jacrev", "jit"]

_P = ParamSpec("_P")


@overload
def jit(
    fun: Literal[None] = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[_P, Any]], stages.JaCeWrapped[_P]]: ...


@overload
def jit(
    fun: Callable[_P, Any],
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> stages.JaCeWrapped[_P]: ...


def jit(
    fun: Callable[_P, Any] | None = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[_P, Any]], stages.JaCeWrapped[_P]] | stages.JaCeWrapped[_P]:
    """
    JaCe's replacement for `jax.jit` (just-in-time) wrapper.

    It works the same way as `jax.jit` does, but instead of lowering the
    computation to XLA, it is lowered to DaCe.
    The function supports a subset of the arguments that are accepted by `jax.jit()`,
    currently none, and some JaCe specific ones.

    Args:
        fun: Function to wrap.
        primitive_translators: Use these primitive translators for the lowering to SDFG.
            If not specified the translators in the global registry are used.
        kwargs: Jit arguments.

    Note:
        This function is the only valid way to obtain a JaCe computation.
    """
    if kwargs:
        # TODO(phimuell): Add proper name verification and exception type.
        raise NotImplementedError(
            f"The following arguments to 'jace.jit' are not yet supported: {', '.join(kwargs)}."
        )

    def wrapper(f: Callable[_P, Any]) -> stages.JaCeWrapped[_P]:
        if any(
            param.default is not param.empty for param in inspect.signature(f).parameters.values()
        ):
            raise NotImplementedError("Default values are not yet supported.")

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
