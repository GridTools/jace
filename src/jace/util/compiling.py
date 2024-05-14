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

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import dace

from jace.translator import post_translation as ptrans
from jace.util import dace_helper as jdace


def compile_jax_sdfg(
    jsdfg: ptrans.FinalizedJaxprSDFG,
    cache: bool = True,
) -> jdace.CompiledSDFG:
    """This function compiles the sdfg embedded in the `FinalizedJaxpr` object and returns it.

    By default the function will store the resulting `CompiledSDFG` object inside `jsdfg` (`FinalizedJaxprSDFG`).
    However, by setting `cache` to `False` the respective field will not be modified.

    Notes:
        Currently the SDFG must not have any undefined symbols, i.e. no undefined sizes.
    """
    from time import time

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

    # To ensure that the SDFG is compiled and to get rid of a warning we must modify
    #  some settings of the SDFG. To fake an immutable SDFG, we will restore them later.
    sdfg: dace.SDFG = jsdfg.sdfg
    org_sdfg_name: str = sdfg.name
    org_recompile: bool = sdfg._recompile
    org_regenerate_code: bool = sdfg._regenerate_code

    try:
        # We need to give the SDFG another name, this is needed to prevent a DaCe error/warning.
        #  This happens if we compile the same lowered SDFG multiple times with different options.
        sdfg.name = f"{sdfg.name}__comp_{int(time() * 1000)}"

        # Actual compiling the stuff; forcing that a recompilation happens
        with dace.config.temporary_config():
            sdfg._recompile = True
            sdfg._regenerate_code = True
            dace.Config.set("compiler", "use_cache", value=False)
            csdfg: jdace.CompiledSDFG = sdfg.compile()

    finally:
        sdfg.name = org_sdfg_name
        sdfg._recompile = org_recompile
        sdfg._regenerate_code = org_regenerate_code

    # Storing the compiled SDFG for later use.
    if cache:
        jsdfg.csdfg = csdfg

    return csdfg


@singledispatch
def run_jax_sdfg(
    csdfg: jdace.CompiledSDFG,
    inp_names: Sequence[str],
    out_names: Sequence[str],
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, ...] | Any:
    """Run the compiled SDFG.

    The function assumes that the `(csdfg, inp_names, out_names)` together form a `FinalizedJaxprSDFG` object.
    Further, it assumes that it was compiled according to this rule.

    Notes:
        This function is used for debugging purposes and you should use the `jace.jit` annotation instead.
        The function assumes that the SDFG was compiled in accordance with `compile_jax_sdfg()`
    """
    from dace.data import Array, Data, Scalar, make_array_from_descriptor

    if len(inp_names) != len(args):
        raise RuntimeError("Wrong number of arguments.")
    if len(kwargs) != 0:
        raise NotImplementedError("No kwargs are supported yet.")

    # We need the SDFG to construct/allocate the memory for the return values.
    #  Actually, we would only need the descriptors, but this is currently the only way to get them.
    # Note that this is save to do, under the assumption that the SDFG, which is inside the CompiledSDFG is still accurate.
    #  But since it is by assumption finalized we should be fine.
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


@run_jax_sdfg.register(ptrans.FinalizedJaxprSDFG)
def _(
    jsdfg: ptrans.FinalizedJaxprSDFG,
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, ...] | Any:
    """Execute the `FinalizedJaxprSDFG` object.

    If `jsdfg` does not have an embedded `CompiledSDFG` already the function will compile it first.
    However, it will not modify the field.
    """

    if jsdfg.csdfg is None:
        csdfg: jdace.CompiledSDFG = compile_jax_sdfg(jsdfg, cache=False)
    else:
        csdfg = jsdfg.csdfg
    return run_jax_sdfg(
        csdfg=csdfg,
        inp_names=jsdfg.inp_names,
        out_names=jsdfg.out_names,
        *args,  # noqa: B026  # star expansion.
        **kwargs,
    )
