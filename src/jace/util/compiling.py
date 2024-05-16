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

import functools as ft
import time
from collections.abc import Mapping, Sequence
from typing import Any

import dace

from jace import translator
from jace.util import dace_helper as jdace


def compile_jax_sdfg(
    tsdfg: translator.TranslatedJaxprSDFG,
) -> jdace.CompiledSDFG:
    """This function compiles the SDFG embedded in the embedded `tsdfg` (`TranslatedJaxprSDFG`).

    Notes:
        Currently the SDFG must not have any undefined symbols, i.e. no undefined sizes.
    """
    if not tsdfg.is_finalized:
        raise ValueError("Can only compile a finalized SDFG.")
    if not tsdfg.inp_names:
        raise ValueError("The passed SDFG did not had any input arguments.")
    if not tsdfg.out_names:
        raise ValueError("The passed SDFG did not had any output arguments.")

    # This is a simplification that makes our life simply.
    #  However, we should consider lifting it at some point.
    if len(tsdfg.sdfg.free_symbols) != 0:
        raise NotImplementedError(
            f"No externally defined symbols are allowed, found: {tsdfg.sdfg.free_symbols}"
        )

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


@ft.singledispatch
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
    """
    from dace.data import Array, Data, Scalar, make_array_from_descriptor

    if len(ckwargs) != 0:
        raise NotImplementedError("No kwargs are supported yet.")
    if len(inp_names) != len(cargs):
        raise RuntimeError("Wrong number of arguments.")

    # We need the SDFG to construct/allocate the memory for the return values.
    #  Actually, we would only need the descriptors, but this is currently the only way to get them.
    #  As far as I know the dace performs a deepcopy before compilation, thus it should be safe.
    #  However, regardless of this this also works if we are inside the stages, which have exclusive ownership.
    sdfg: dace.SDFG = csdfg.sdfg

    # Build the argument list that we will pass to the compiled object.
    call_args: dict[str, Any] = {}
    for in_name, in_val in zip(inp_names, cargs, strict=True):
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


@run_jax_sdfg.register(translator.TranslatedJaxprSDFG)
def _(
    tsdfg: translator.TranslatedJaxprSDFG,
    cargs: Sequence[Any],
    ckwargs: Mapping[str, Any],
) -> tuple[Any, ...] | Any:
    """Execute the `TranslatedJaxprSDFG` object directly.

    This function is a convenience function provided for debugging.
    """
    csdfg: jdace.CompiledSDFG = compile_jax_sdfg(tsdfg)
    return run_jax_sdfg(
        csdfg=csdfg,
        inp_names=tsdfg.inp_names,
        out_names=tsdfg.out_names,
        cargs=cargs,
        ckwargs=ckwargs,
    )
