# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jax.*` namespace."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Literal, ParamSpec, TypedDict, TypeVar, overload

from jax import grad, jacfwd, jacrev
from typing_extensions import Unpack

from jace import stages, translator


__all__ = ["JITOptions", "grad", "jacfwd", "jacrev", "jit"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


class JITOptions(TypedDict, total=False):
    """
    All known options to `jace.jit` that influence tracing.

    Not all arguments that are supported by `jax-jit()` are also supported by
    `jace.jit`. Furthermore, some additional ones might be supported.
    The following arguments are supported:
    - `backend`: For which platform DaCe should generate code for. It is a string,
        where the following values are supported: `'cpu'` or `'gpu'`.
        DaCe's `DeviceType` enum or FPGA are not supported.
    """

    backend: str


@overload
def jit(
    fun: Literal[None] = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JITOptions],
) -> Callable[[Callable[_P, _R]], stages.JaCeWrapped[_P, _R]]: ...


@overload
def jit(
    fun: Callable[_P, _R],
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JITOptions],
) -> stages.JaCeWrapped[_P, _R]: ...


def jit(
    fun: Callable[_P, _R] | None = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JITOptions],
) -> Callable[[Callable[_P, _R]], stages.JaCeWrapped[_P, _R]] | stages.JaCeWrapped[_P, _R]:
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
        kwargs: Jit arguments, see `JITOptions` for more.

    Note:
        This function is the only valid way to obtain a JaCe computation.
    """
    not_supported_jit_keys = kwargs.keys() - {"backend"}
    if not_supported_jit_keys:
        # TODO(phimuell): Add proper name verification and exception type.
        raise NotImplementedError(
            f"The following arguments to 'jace.jit' are not yet supported: {', '.join(not_supported_jit_keys)}."
        )

    def wrapper(f: Callable[_P, _R]) -> stages.JaCeWrapped[_P, _R]:
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
