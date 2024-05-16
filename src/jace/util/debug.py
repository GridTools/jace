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

from collections.abc import Callable
from typing import Any

import dace
import jax

from jace import translator


def run_jax_sdfg(jsdfg: translator.TranslatedJaxprSDFG, *args: Any) -> tuple[Any, ...] | Any:
    """Calls the SDFG that is encapsulated with the supplied arguments.

    Notes:
        Currently the SDFG must not have any undefined symbols, i.e. no undefined sizes.
        Currently denoted arguments are not fully respected.
        The function either returns a value or a tuple of values, i.e. no tree.
    """
    from dace.data import Array, Data, Scalar, make_array_from_descriptor

    # This is a simplification that makes our life simply
    if len(jsdfg.sdfg.free_symbols) != 0:
        raise ValueError(
            f"No externally defined symbols are allowed, found: {jsdfg.sdfg.free_symbols}"
        )
    if len(jsdfg.inp_names) != len(args):
        raise ValueError(
            f"Wrong numbers of arguments expected {len(jsdfg.inp_names)} got {len(args)}."
        )

    # We use a return by reference approach, for calling the SDFG
    call_args: dict[str, Any] = {}
    for in_name, in_val in zip(jsdfg.inp_names, args):
        call_args[in_name] = in_val
    for out_name in jsdfg.out_names:
        sarray: Data = jsdfg.sdfg.arrays[out_name]
        assert out_name not in call_args

        if (out_name == "__return") or (out_name.startswith("__return_")):
            continue
        if isinstance(sarray, Scalar):
            raise NotImplementedError("Scalars as return values are not supported.")
        if isinstance(sarray, Array):
            call_args[out_name] = make_array_from_descriptor(sarray)
        else:
            raise NotImplementedError(f"Can not handle '{type(sarray).__name__}' as output.")

    # Canonical SDFGs do not have global memory, so we must transform it.
    #  We will afterwards undo it.
    for glob_name in jsdfg.inp_names + jsdfg.out_names:
        jsdfg.sdfg.arrays[glob_name].transient = False

    try:
        csdfg: dace.CompiledSDFG = jsdfg.sdfg.compile()
        with dace.config.temporary_config():
            dace.Config.set("compiler", "allow_view_arguments", value=True)
            csdfg(**call_args)

        if len(jsdfg.out_names) == 0:
            return None
        ret_val: tuple[Any] = tuple(call_args[out_name] for out_name in jsdfg.out_names)
        if len(jsdfg.out_names) == 1:
            return ret_val[0]
        return ret_val

    finally:
        for name in jsdfg.inp_names + jsdfg.out_names:
            jsdfg.sdfg.arrays[name].transient = True


def _jace_run(fun: Callable, *args: Any) -> Any:
    """Traces and run function `fun` using `Jax | DaCe`.

    Args:
        *args:      Forwarded to the tracing and final execution of the SDFG.

    Notes:
        This function will be removed soon.
    """
    jaxpr = jax.make_jaxpr(fun)(*args)
    driver = translator.JaxprTranslationDriver(translator.get_subtranslators())
    jsdfg = driver.translate_jaxpr(jaxpr)
    return run_jax_sdfg(jsdfg, *args)
