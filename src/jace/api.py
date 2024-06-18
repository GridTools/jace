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
from typing import TYPE_CHECKING, Literal, ParamSpec, TypedDict, TypeVar, overload

from jax import grad, jacfwd, jacrev
from typing_extensions import Unpack

from jace import stages, translator


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


__all__ = ["JitOptions", "grad", "jacfwd", "jacrev", "jit"]

# Used for type annotation, see the notes in `jace.stages` for more.
_P = ParamSpec("_P")
_RetrunType = TypeVar("_RetrunType")


class JitOptions(TypedDict, total=False):
    """
    All known options to `jace.jit` that influence tracing.

    Notes:
        Currently there are no known options, but essentially it is a subset of some
        of the options that are supported by `jax.jit` together with some additional
        JaCe specific ones.
    """


@overload
def jit(
    fun: Literal[None] = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JitOptions],
) -> Callable[[Callable[_P, _RetrunType]], stages.JaCeWrapped[_P, _RetrunType]]: ...


@overload
def jit(
    fun: Callable[_P, _RetrunType],
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JitOptions],
) -> stages.JaCeWrapped[_P, _RetrunType]: ...


def jit(
    fun: Callable[_P, _RetrunType] | None = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JitOptions],
) -> (
    Callable[[Callable[_P, _RetrunType]], stages.JaCeWrapped[_P, _RetrunType]]
    | stages.JaCeWrapped[_P, _RetrunType]
):
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

    def wrapper(f: Callable[_P, _RetrunType]) -> stages.JaCeWrapped[_P, _RetrunType]:
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
