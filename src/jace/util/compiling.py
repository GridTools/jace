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

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import dace
import numpy as np
from dace import data as ddata


if TYPE_CHECKING:
    from jace import translator
    from jace.util import dace_helper as jdace


def compile_jax_sdfg(
    tsdfg: translator.TranslatedJaxprSDFG,
) -> jdace.CompiledSDFG:
    """This function compiles the SDFG embedded in the embedded `tsdfg` (`TranslatedJaxprSDFG`).

    For executing the SDFG, the `run_jax_sdfg()` function, together with the `tsdfg.{inp, out}_names` can be used.
    """
    if not tsdfg.is_finalized:
        raise ValueError("Can only compile a finalized SDFG.")
    if any(  # We do not support the DaCe return mechanism
        arrname.startswith("__return")
        for arrname in tsdfg.sdfg.arrays.keys()  # noqa: SIM118  # we can not use `in` because we are also interested in `__return_`!
    ):
        raise ValueError("Only support SDFGs without '__return' members.")

    # To ensure that the SDFG is compiled and to get rid of a warning we must modify
    #  some settings of the SDFG. To fake an immutable SDFG, we will restore them later.
    sdfg = tsdfg.sdfg
    org_sdfg_name = sdfg.name
    org_recompile = sdfg._recompile
    org_regenerate_code = sdfg._regenerate_code

    try:
        # We need to give the SDFG another name, this is needed to prevent a DaCe error/warning.
        #  This happens if we compile the same lowered SDFG multiple times with different options.
        sdfg.name = f"{sdfg.name}__comp_{int(time.time() * 1000)}"

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

    return csdfg


def run_jax_sdfg(
    csdfg: jdace.CompiledSDFG,
    inp_names: Sequence[str],
    out_names: Sequence[str],
    cargs: Sequence[Any],
    ckwargs: Mapping[str, Any],
) -> tuple[Any, ...] | Any:
    """Run the compiled SDFG.

    The function assumes that the SDFG was finalized and then compiled by `compile_jax_sdfg()`.

    Args:
        csdfg:      The `CompiledSDFG` object.
        inp_names:  List of names of the input arguments.
        out_names:  List of names of the output arguments.
        cargs:      All positional arguments of the call.
        ckwargs:    All keyword arguments of the call.

    Notes:
        There is no pytree mechanism jet, thus the return values are returned inside a `tuple`
            or in case of one value, directly, in the order determined by Jax.
        Currently, this function does not consider strides in the input,
            all input must be `C_CONTIGUOUS`.
        Currently the SDFG must not have any undefined symbols, i.e. no undefined sizes.
    """
    from jace import util

    sdfg: dace.SDFG = csdfg.sdfg

    if len(ckwargs) != 0:
        raise NotImplementedError("No kwargs are supported yet.")
    if len(inp_names) != len(cargs):
        raise RuntimeError("Wrong number of arguments.")
    if len(set(inp_names).intersection(out_names)) != 0:
        raise NotImplementedError("Using an input also for output is not yet supported.")
    if len(sdfg.free_symbols) != 0:  # This is a simplification that makes our life simple.
        raise NotImplementedError(
            f"No externally defined symbols are allowed, found: {sdfg.free_symbols}"
        )

    # Build the argument list that we will pass to the compiled object.
    call_args: dict[str, Any] = {}
    for in_name, in_val in zip(inp_names, cargs, strict=True):
        if util.is_scalar(in_val):
            # Currently the translator makes scalar into arrays, this has to be reflected here
            in_val = np.array([in_val])
        call_args[in_name] = in_val

    for out_name, sarray in ((name, sdfg.arrays[name]) for name in out_names):
        assert not (out_name in call_args and util.is_jax_array(call_args[out_name]))
        assert isinstance(sarray, ddata.Array)
        call_args[out_name] = ddata.make_array_from_descriptor(sarray)

    assert len(call_args) == len(csdfg.argnames), (
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
