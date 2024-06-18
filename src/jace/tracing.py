# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Implements the tracing machinery that is used to build the Jaxpr.

Essentially, Jax provides `jax.make_jaxpr()` which is essentially a debug utility. Jax
does not provide any public way to get a Jaxpr. This module provides the necessary
functionality for use in JaCe.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, overload

import jax
from jax import tree_util as jax_tree


if TYPE_CHECKING:
    from collections.abc import Callable

    from jace import api

_P = ParamSpec("_P")
_RetrunType = TypeVar("_RetrunType")


@overload
def make_jaxpr(
    fun: Callable[_P, _RetrunType],
    trace_options: api.JitOptions,
    return_outtree: Literal[True],
) -> Callable[_P, tuple[jax.core.ClosedJaxpr, jax_tree.PyTreeDef]]: ...


@overload
def make_jaxpr(
    fun: Callable[_P, _RetrunType],
    trace_options: api.JitOptions,
    return_outtree: Literal[False] = False,
) -> Callable[_P, jax.core.ClosedJaxpr]: ...


def make_jaxpr(
    fun: Callable[_P, Any],
    trace_options: api.JitOptions,
    return_outtree: bool = False,
) -> (
    Callable[_P, tuple[jax.core.ClosedJaxpr, jax_tree.PyTreeDef]]
    | Callable[_P, jax.core.ClosedJaxpr]
):
    """
    JaCe's replacement for `jax.make_jaxpr()`.

    Returns a callable object that produces as Jaxpr and optionally a pytree defining
    the output. By default the callable will only return the Jaxpr, however, by setting
    `return_outtree` the function will also return the output tree, this is different
    from the `return_shape` of `jax.make_jaxpr()`.

    Currently the tracing is always performed with an enabled `x64` mode.

    Returns:
        The function returns a callable, that if passed arguments will performs the
        tracing on them, this section will describe the return value of that function.
        If `return_outtree` is `False` the function will simply return the generated
        Jaxpr. If `return_outtree` is `True` the function will return a pair.
        The first element is the Jaxpr and the second element is a pytree object
        that describes the output.

    Args:
        fun: The original Python computation.
        trace_options: The options used for tracing, the same arguments that
            are supported by `jace.jit`.
        return_outtree: Also return the pytree of the output.

    Todo:
        - Handle default arguments of `fun`.
        - Handle static arguments.
        - Turn `trace_options` into a `TypedDict` and sync with `jace.jit`.
    """
    if trace_options:
        raise NotImplementedError(
            f"Not supported tracing options: {', '.join(f'{k}' for k in trace_options)}"
        )
    assert all(param.default is param.empty for param in inspect.signature(fun).parameters.values())

    def tracer_impl(
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> tuple[jax.core.ClosedJaxpr, jax_tree.PyTreeDef] | jax.core.ClosedJaxpr:
        # In Jax `float32` is the main datatype, and they go to great lengths to avoid
        #  some aggressive [type promotion](https://jax.readthedocs.io/en/latest/type_promotion.html).
        #  However, in this case we will have problems when we call the SDFG, for some
        #  reasons `CompiledSDFG` does not work in that case correctly, thus we enable
        #  it for the tracing.
        with jax.experimental.enable_x64():
            # TODO(phimuell): copy the implementation of the real tracing
            jaxpr_maker = jax.make_jaxpr(
                fun,
                **trace_options,
                return_shape=True,
            )
            jaxpr, outshapes = jaxpr_maker(
                *args,
                **kwargs,
            )

        if not return_outtree:
            return jaxpr

        # Regardless what the documentation of `make_jaxpr` claims, it does not output
        #  a pytree instead an abstract description of the shape, that we will
        #  transform into a pytree.
        outtree = jax_tree.tree_structure(outshapes)
        return jaxpr, outtree

    return tracer_impl  # type: ignore[return-value]  # Type confusion
