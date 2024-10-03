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
from typing import Any, Final, Literal, ParamSpec, TypedDict, overload

from jax import grad, jacfwd, jacrev
from typing_extensions import Unpack

from jace import stages, translator, util


__all__ = ["DEFAUL_BACKEND", "JITOptions", "grad", "jacfwd", "jacrev", "jit"]

_P = ParamSpec("_P")

DEFAUL_BACKEND: Final[str] = "cpu"


class JITOptions(TypedDict, total=False):
    """
    All known options to `jace.jit` that influence tracing.

    Not all arguments that are supported by `jax-jit()` are also supported by
    `jace.jit`. Furthermore, some additional ones might be supported.

    Args:
    backend: Target platform for which DaCe should generate code. Supported values
        are `'cpu'` or `'gpu'`.
    """

    backend: str


@overload
def jit(
    fun: Literal[None] = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JITOptions],
) -> Callable[[Callable[_P, Any]], stages.JaCeWrapped[_P]]: ...


@overload
def jit(
    fun: Callable[_P, Any],
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JITOptions],
) -> stages.JaCeWrapped[_P]: ...


def jit(
    fun: Callable[_P, Any] | None = None,
    /,
    primitive_translators: Mapping[str, translator.PrimitiveTranslator] | None = None,
    **kwargs: Unpack[JITOptions],
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
        kwargs: JIT arguments, see `JITOptions` for more.

    Note:
        This function is the only valid way to obtain a JaCe computation.
    """
    not_supported_jit_keys = kwargs.keys() - JITOptions.__annotations__.keys()
    if not_supported_jit_keys:
        raise ValueError(
            f"The following arguments to 'jace.jit' are not supported: {', '.join(not_supported_jit_keys)}."
        )
    if kwargs.get("backend", DEFAUL_BACKEND).lower() not in {"cpu", "gpu"}:
        raise ValueError(f"The backend '{kwargs['backend']}' is not supported.")

    def wrapper(f: Callable[_P, Any]) -> stages.JaCeWrapped[_P]:
        jace_wrapper = stages.JaCeWrapped(
            fun=f,
            primitive_translators=(
                translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY
                if primitive_translators is None
                else primitive_translators
            ),
            jit_options=kwargs,
            device=util.parse_backend_jit_option(kwargs.get("backend", DEFAUL_BACKEND)),
        )
        functools.update_wrapper(jace_wrapper, f)
        return jace_wrapper

    return wrapper if fun is None else wrapper(fun)
