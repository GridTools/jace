# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements all utility functions that are related to DaCe."""

from __future__ import annotations

import os
import pathlib
import time
from typing import TYPE_CHECKING, Any

import dace
from dace import data as dace_data
from dace.codegen.compiled_sdfg import CompiledSDFG
from jax import tree_util as jax_tree

from jace import util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jace import translator
    from jace.util import dace_helper

__all__ = ["CompiledSDFG", "compile_jax_sdfg", "run_jax_sdfg"]


def compile_jax_sdfg(tsdfg: translator.TranslatedJaxprSDFG) -> dace_helper.CompiledSDFG:
    """Compiles the SDFG embedded in `tsdfg` and return the resulting `CompiledSDFG` object."""
    if any(  # We do not support the DaCe return mechanism
        array_name.startswith("__return")
        for array_name in tsdfg.sdfg.arrays.keys()  # noqa: SIM118  # We can not use `in` because we are not interested in `my_mangled_variable__return_zulu`!
    ):
        raise ValueError("Only support SDFGs without '__return' members.")
    if tsdfg.sdfg.free_symbols:  # This is a simplification that makes our life simple.
        raise NotImplementedError(f"No free symbols allowed, found: {tsdfg.sdfg.free_symbols}")

    # To ensure that the SDFG is compiled and to get rid of a warning we must modify
    #  some settings of the SDFG. But we also have to fake an immutable SDFG
    sdfg = tsdfg.sdfg
    org_sdfg_name = sdfg.name
    org_recompile = sdfg._recompile
    org_regenerate_code = sdfg._regenerate_code

    try:
        # We need to give the SDFG another name, this is needed to prevent a DaCe error/warning.
        #  This happens if we compile the same lowered SDFG multiple times with different options.
        sdfg.name = f"{sdfg.name}__comp_{int(time.time() * 1000)}_{os.getpid()}"
        assert len(sdfg.name) < 255

        with dace.config.temporary_config():
            dace.Config.set("compiler", "use_cache", value=False)
            dace.Config.set("cache", value="name")
            dace.Config.set("default_build_folder", value=pathlib.Path(".jacecache").resolve())
            sdfg._recompile = True
            sdfg._regenerate_code = True
            csdfg: dace_helper.CompiledSDFG = sdfg.compile()

    finally:
        sdfg.name = org_sdfg_name
        sdfg._recompile = org_recompile
        sdfg._regenerate_code = org_regenerate_code

    return csdfg


def run_jax_sdfg(
    csdfg: dace_helper.CompiledSDFG,
    inp_names: Sequence[str],
    out_names: Sequence[str],
    flat_call_args: Sequence[Any],
    outtree: jax_tree.PyTreeDef,
) -> tuple[Any, ...] | Any:
    """Run the compiled SDFG.

    The function assumes that the SDFG was finalized and then compiled by
    `compile_jax_sdfg()`. For running the SDFG you also have to pass the input
    names (`inp_names`) and output names (`out_names`) that were inside the
    `TranslatedJaxprSDFG` from which `csdfg` was compiled from.

    Args:
        csdfg: The `CompiledSDFG` object.
        inp_names: List of names of the input arguments.
        out_names: List of names of the output arguments.
        flat_call_args: Flattened input arguments.
        outtree: A pytree describing how to unflatten the output.

    Notes:
        Currently the strides of the input arguments must match the ones that
        were used for lowering.
    """
    if len(inp_names) != len(flat_call_args):
        raise RuntimeError("Wrong number of arguments.")

    sdfg_call_args: dict[str, Any] = {}
    for in_name, in_val in zip(inp_names, flat_call_args, strict=True):
        # TODO(phimuell): Implement a stride matching process.
        if util.is_jax_array(in_val):
            if not util.is_fully_addressable(in_val):
                raise ValueError(f"Passed a not fully addressable Jax array as '{in_name}'")
            in_val = in_val.__array__()
        sdfg_call_args[in_name] = in_val

    arrays = csdfg.sdfg.arrays
    for out_name, sdfg_array in ((out_name, arrays[out_name]) for out_name in out_names):
        if out_name in sdfg_call_args:
            if util.is_jax_array(sdfg_call_args[out_name]):
                raise ValueError("Passed an immutable Jax array as output.")
        else:
            sdfg_call_args[out_name] = dace_data.make_array_from_descriptor(sdfg_array)

    assert len(sdfg_call_args) == len(csdfg.argnames), (
        "Failed to construct the call arguments,"
        f" expected {len(csdfg.argnames)} but got {len(flat_call_args)}."
        f"\nExpected: {csdfg.argnames}\nGot: {list(sdfg_call_args.keys())}"
    )

    # Calling the SDFG
    with dace.config.temporary_config():
        dace.Config.set("compiler", "allow_view_arguments", value=True)
        csdfg(**sdfg_call_args)

    if not out_names:
        return None
    return jax_tree.tree_unflatten(outtree, (sdfg_call_args[out_name] for out_name in out_names))
