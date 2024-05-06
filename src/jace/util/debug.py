# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains functions for debugging the translator.

Everything in this module is experimental and might vanish anytime.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import Any

import dace
import jax

from jace import translator


def compile_jax_sdfg(
    jsdfg: translator.TranslatedJaxprSDFG, force: bool = False, save: bool = True
) -> dace.CompiledSDFG:
    """This function compiles the embedded SDFG and return it.

    The SDFG is compiled in a very special way, i.e. all arguments and return values have to be passed as arguments.

    Before doing anything the function will inspect the `csdfg` filed of the `TranslatedJaxprSDFG`.
    If it is not `None` the function will return this value.
    This can be disabled by setting `focre` to `True`.
    If the SDFG is compiled the function will store the compiled SDFG inside the `TranslatedJaxprSDFG` object's `csdfg` field.
    However, by setting `save` to `False` the field will not be modified.

    Args:
        force:      Force compilation even if the `csdfg` field is already set.
        save:       Store the compiled SDFG inside the `TranslatedJaxprSDFG` object's `csdfg` field.

    Notes:
        Currently the SDFG must not have any undefined symbols, i.e. no undefined sizes.
        The function either returns a value or a tuple of values, i.e. no tree.
    """
    if not jsdfg.inp_names:
        raise ValueError("The passed SDFG did not had any input arguments.")
    if not jsdfg.out_names:
        raise ValueError("The passed SDFG did not had any output arguments.")
    if any(out_name.startswith("__return") for out_name in jsdfg.out_names):
        raise NotImplementedError("No return statement is supported yet.")

    if (not force) and (jsdfg.csdfg is not None):
        assert isinstance(jsdfg.csdfg, dace.CompiledSDFG)
        return jsdfg.csdfg

    # This is a simplification that makes our life simply.
    #  However, we should consider lifting it at some point.
    if len(jsdfg.sdfg.free_symbols) != 0:
        raise ValueError(
            f"No externally defined symbols are allowed, found: {jsdfg.sdfg.free_symbols}"
        )

    # Canonical SDFGs do not have global memory, so we must transform it; undo afterwards
    prev_trans_state: dict[str, bool] = {}
    for glob_name in jsdfg.inp_names + jsdfg.out_names:  # type: ignore[operator]  # concatenation
        if glob_name in prev_trans_state:  # Donated arguments
            continue
        prev_trans_state[glob_name] = jsdfg.sdfg.arrays[glob_name].transient
        jsdfg.sdfg.arrays[glob_name].transient = False

    try:
        csdfg: dace.CompiledSDFG = jsdfg.sdfg.compile()
        if save:
            jsdfg.csdfg = csdfg
        return csdfg

    finally:
        # Restore the initial transient state
        for var_name, trans_state in prev_trans_state.items():
            jsdfg.sdfg.arrays[var_name].transient = trans_state


@singledispatch
def run_jax_sdfg(
    jsdfg: translator.TranslatedJaxprSDFG,
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, ...] | Any:
    """Run the `TranslatedJaxprSDFG` object.

    If the `TranslatedJaxprSDFG` object does not contain a precompiled SDFG object the function will compile it.
    However, the compiled SDFG will not be cached in the `TranslatedJaxprSDFG` object.
    """
    if jsdfg.inp_names is None:
        raise ValueError("Input names are not specified.")
    if jsdfg.out_names is None:
        raise ValueError("Output names are not specified.")

    if jsdfg.csdfg is not None:
        csdfg: dace.CompiledSDFG = jsdfg.csdfg
    else:
        csdfg = compile_jax_sdfg(jsdfg, save=False)
    return run_jax_sdfg(
        csdfg,
        jsdfg.inp_names,
        jsdfg.out_names,
        *args,
        **kwargs,
    )


@run_jax_sdfg.register(dace.CompiledSDFG)
def _(
    csdfg: dace.CompiledSDFG,
    inp_names: Sequence[str],
    out_names: Sequence[str],
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, ...] | Any:
    """Call the compiled SDFG.

    The function assumes that the SDFG was compiled in accordance with `compile_jax_sdfg()`
    """
    from dace.data import Array, Data, Scalar, make_array_from_descriptor

    if len(inp_names) != len(args):
        raise RuntimeError("Wrong number of arguments.")
    if len(kwargs) != 0:
        raise NotImplementedError("No kwargs are supported yet.")

    sdfg: dace.SDFG = csdfg.sdfg

    # Build the argument list that we will pass to the compiled object.
    call_args: dict[str, Any] = {}
    for in_name, in_val in zip(inp_names, args):
        call_args[in_name] = in_val
    for out_name in out_names:
        assert not ((out_name == "__return") or (out_name.startswith("__return_")))  # noqa: PT018 # Assert split

        if out_name in call_args:  # Donated arguments
            assert out_name in inp_names
            continue

        sarray: Data = sdfg.arrays[out_name]
        if isinstance(sarray, Scalar):
            raise NotImplementedError("Scalars as return values are not supported.")
        if isinstance(sarray, Array):
            call_args[out_name] = make_array_from_descriptor(sarray)
        else:
            raise NotImplementedError(f"Can not handle '{type(sarray).__name__}' as output.")

    if len(call_args) != len(csdfg.argnames):
        raise ValueError(
            "Failed to construct the call arguments,"
            f" expected {len(csdfg.argnames)} but got {len(call_args)}."
        )

    # Calling the SDFG
    with dace.config.temporary_config():
        dace.Config.set("compiler", "allow_view_arguments", value=True)
        csdfg(**call_args)

    # Handling the output (pytrees are missing)
    if len(out_names) == 0:
        return None
    ret_val: tuple[Any] = tuple(call_args[out_name] for out_name in out_names)
    if len(out_names) == 1:
        return ret_val[0]
    return ret_val


def _jace_run(
    fun: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Traces and run function `fun` using `Jax | DaCe`.

    Args:
        *args:      Forwarded to the tracing and final execution of the SDFG.
        **kwargs:   Used to construct the driver.
    """
    jaxpr = jax.make_jaxpr(fun)(*args)
    driver = translator.JaxprTranslationDriver(**kwargs)
    jsdfg = driver.translate_jaxpr(jaxpr)
    return run_jax_sdfg(jsdfg, *args)
