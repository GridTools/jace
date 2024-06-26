# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements all utility functions that are related to DaCe."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import dace
import numpy as np
from dace import data as dace_data

# The compiled SDFG is not available in the dace namespace or anywhere else
#  Thus we import it here directly
from dace.codegen.compiled_sdfg import CompiledSDFG

from jace import util


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jace import translator

__all__ = ["CompiledSDFG", "compile_jax_sdfg", "run_jax_sdfg"]


def compile_jax_sdfg(tsdfg: translator.TranslatedJaxprSDFG) -> CompiledSDFG:
    """Compiles the embedded SDFG and return the resulting `CompiledSDFG` object."""
    if any(  # We do not support the DaCe return mechanism
        array_name.startswith("__return")
        for array_name in tsdfg.sdfg.arrays.keys()  # noqa: SIM118  # we can not use `in` because we are also interested in `__return_`!
    ):
        raise ValueError("Only support SDFGs without '__return' members.")

    # To ensure that the SDFG is compiled and to get rid of a warning we must modify
    #  some settings of the SDFG. To fake an immutable SDFG, we will restore them later.
    sdfg = tsdfg.sdfg
    org_sdfg_name = sdfg.name
    org_recompile = sdfg._recompile
    org_regenerate_code = sdfg._regenerate_code

    try:
        # We need to give the SDFG another name, this is needed to prevent a DaCe
        #  error/warning. This happens if we compile the same lowered SDFG multiple
        #  times with different options.
        sdfg.name = f"{sdfg.name}__comp_{int(time.time() * 1000)}"

        with dace.config.temporary_config():
            sdfg._recompile = True
            sdfg._regenerate_code = True
            dace.Config.set("compiler", "use_cache", value=False)
            csdfg: CompiledSDFG = sdfg.compile()

    finally:
        sdfg.name = org_sdfg_name
        sdfg._recompile = org_recompile
        sdfg._regenerate_code = org_regenerate_code

    return csdfg


def run_jax_sdfg(
    csdfg: CompiledSDFG,
    inp_names: Sequence[str],
    out_names: Sequence[str],
    call_args: Sequence[Any],
    call_kwargs: Mapping[str, Any],
) -> tuple[Any, ...] | Any:
    """
    Run the compiled SDFG.

    The function assumes that the SDFG was finalized and then compiled by
    `compile_jax_sdfg()`. For running the SDFG you also have to pass the input
    names (`inp_names`) and output names (`out_names`) that were inside the
    `TranslatedJaxprSDFG` from which `csdfg` was compiled from.

    Args:
        csdfg: The `CompiledSDFG` object.
        inp_names: List of names of the input arguments.
        out_names: List of names of the output arguments.
        call_args: All positional arguments of the call.
        call_kwargs: All keyword arguments of the call.

    Note:
        There is no pytree mechanism jet, thus the return values are returned
        inside a `tuple` or in case of one value, directly, in the order
        determined by Jax. Furthermore, DaCe does not support scalar return
        values, thus they are silently converted into arrays of length 1, the
        same holds for inputs.

    Todo:
        - Implement non C strides.
    """
    sdfg: dace.SDFG = csdfg.sdfg

    if len(call_kwargs) != 0:
        raise NotImplementedError("No kwargs are supported yet.")
    if len(inp_names) != len(call_args):
        raise RuntimeError("Wrong number of arguments.")
    if sdfg.free_symbols:  # This is a simplification that makes our life simple.
        raise NotImplementedError(
            f"No externally defined symbols are allowed, found: {sdfg.free_symbols}"
        )

    # Build the argument list that we will pass to the compiled object.
    sdfg_call_args: dict[str, Any] = {}
    for in_name, in_val in zip(inp_names, call_args, strict=True):
        if util.is_scalar(in_val):
            # Currently the translator makes scalar into arrays, this has to be
            #  reflected here
            in_val = np.array([in_val])  # noqa: PLW2901  # Loop variable is intentionally modified.
        sdfg_call_args[in_name] = in_val

    for out_name, sdfg_array in ((out_name, sdfg.arrays[out_name]) for out_name in out_names):
        if out_name in sdfg_call_args:
            if util.is_jax_array(sdfg_call_args[out_name]):
                # Jax arrays are immutable, so they can not be return values too.
                raise ValueError("Passed a Jax array as output.")
        else:
            sdfg_call_args[out_name] = dace_data.make_array_from_descriptor(sdfg_array)

    assert len(sdfg_call_args) == len(csdfg.argnames), (
        "Failed to construct the call arguments,"
        f" expected {len(csdfg.argnames)} but got {len(call_args)}."
        f"\nExpected: {csdfg.argnames}\nGot: {list(sdfg_call_args.keys())}"
    )

    # Calling the SDFG
    with dace.config.temporary_config():
        dace.Config.set("compiler", "allow_view_arguments", value=True)
        csdfg(**sdfg_call_args)

    # Handling the output (pytrees are missing)
    if not out_names:
        return None
    ret_val: tuple[Any] = tuple(sdfg_call_args[out_name] for out_name in out_names)
    return ret_val[0] if len(out_names) == 1 else ret_val
