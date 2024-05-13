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
from jace.util import dace_helper as jdace


def compile_jax_sdfg(
    jsdfg: translator.TranslatedJaxprSDFG,
) -> jdace.CompiledSDFG:
    """This function compiles the embedded SDFG and return it.

    The SDFG is compiled in a very special way, i.e. all arguments and return values have to be passed as arguments.

    Notes:
        Currently the SDFG must not have any undefined symbols, i.e. no undefined sizes.
    """
    from copy import deepcopy

    if not jsdfg.inp_names:
        raise ValueError("The passed SDFG did not had any input arguments.")
    if not jsdfg.out_names:
        raise ValueError("The passed SDFG did not had any output arguments.")
    if any(out_name.startswith("__return") for out_name in jsdfg.out_names):
        raise NotImplementedError("No return statement is supported yet.")

    # This is a simplification that makes our life simply.
    #  However, we should consider lifting it at some point.
    if len(jsdfg.sdfg.free_symbols) != 0:
        raise ValueError(
            f"No externally defined symbols are allowed, found: {jsdfg.sdfg.free_symbols}"
        )

    # We will now deepcopy the SDFG.
    #  We do this because the SDFG is also a member of the `CompiledSDFG` object.
    #  And currently we rely on the integrity of this object in the run function,
    #  i.e. in the allocation of the return values as well as `arg_names`.
    sdfg: dace.SDFG = deepcopy(jsdfg.sdfg)

    # Canonical SDFGs do not have global memory, so we must transform it
    sdfg_arg_names: list[str] = []
    for glob_name in jsdfg.inp_names + jsdfg.out_names:
        if glob_name in sdfg_arg_names:  # Donated arguments
            continue
        sdfg.arrays[glob_name].transient = False
        sdfg_arg_names.append(glob_name)

    # This forces the signature of the SDFG to include all arguments in order they appear.
    sdfg.arg_names = sdfg_arg_names

    # Actual compiling the stuff
    csdfg: jdace.CompiledSDFG = sdfg.compile()
    return csdfg


@singledispatch
def run_jax_sdfg(
    jsdfg: translator.TranslatedJaxprSDFG,
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, ...] | Any:
    """Run the `TranslatedJaxprSDFG` object.

    Notes:
        The function either returns a value or a tuple of values, i.e. no tree.
        There is an overload of this function that accepts an already compiled SDFG and runs it.
    """
    if jsdfg.inp_names is None:
        raise ValueError("Input names are not specified.")
    if jsdfg.out_names is None:
        raise ValueError("Output names are not specified.")
    csdfg: jdace.CompiledSDFG = compile_jax_sdfg(jsdfg)

    return run_jax_sdfg(
        csdfg,
        jsdfg.inp_names,
        jsdfg.out_names,
        *args,
        **kwargs,
    )


@run_jax_sdfg.register(jdace.CompiledSDFG)
def _(
    csdfg: jdace.CompiledSDFG,
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

    # We need the SDFG to construct/allocate the memory for the return values.
    #  Actually, we would only need the descriptors, but this is currently the only way to get them.
    #  Note that this is safe to do, because in the compile function we decoupled the SDFG from all.
    sdfg: dace.SDFG = csdfg.sdfg

    # Build the argument list that we will pass to the compiled object.
    call_args: dict[str, Any] = {}
    for in_name, in_val in zip(inp_names, args, strict=True):
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
            f"\nExpected: {csdfg.argnames}\nGot: {list(call_args.keys())}"
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

    Notes:
        This function will be removed soon.
    """
    jaxpr = jax.make_jaxpr(fun)(*args)
    driver = translator.JaxprTranslationDriver(**kwargs)
    jsdfg = driver.translate_jaxpr(jaxpr)
    return run_jax_sdfg(jsdfg, *args)
