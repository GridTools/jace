# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains functions for debugging the translator."""

from __future__ import annotations

from typing import Any

import dace

from jace.translator import util as jtrutil


def run_memento(
    memento: jtrutil.JaCeTranslationMemento,
    *args: Any,
) -> tuple[Any, ...] | Any:
    """Calls the SDFG with the supplied arguments.

    Notes:
        Currently the SDFG must not have any undefined symbols, i.e. no undefined sizes.
        The function either returns a value or a tuple of values, i.e. no tree.
    """
    from dace.data import Data, Scalar, make_array_from_descriptor

    # This is a simplification that makes our life simply
    if len(memento.sdfg.used_symbols) != 0:
        raise ValueError("No externally defined symbols are allowed.")
    if len(memento.inp_names) != len(args):
        raise ValueError(
            f"Wrong numbers of arguments expected {len(memento.inp_names)} got {len(args)}."
        )

    # We use a return by reference approach, for calling the SDFG
    call_args: dict[str, Any] = {}
    for in_name, in_val in zip(memento.inp_names, args):
        call_args[in_name] = in_val
    for out_name in memento.out_names:
        sarray: Data = memento.sdfg.arrays[out_name]
        if isinstance(sarray, Scalar):
            raise NotImplementedError("Do not support non array in return value.")
        assert out_name not in call_args
        call_args[out_name] = make_array_from_descriptor(sarray)

    # Canonical SDFGs do not have global memory, so we must transform it.
    #  We will afterwards undo it.
    for glob_name in memento.inp_names + memento.out_names:  # type: ignore[operator]  # concatenation
        memento.sdfg.arrays[glob_name].transient = True

    try:
        csdfg: dace.CompiledSDFG = memento.sdfg.compile()
        csdfg(**call_args)

        if len(memento.out_names) == 0:
            return None
        ret_val: tuple[Any] = tuple(call_args[out_name] for out_name in memento.out_names)
        if len(memento.out_names) == 1:
            return ret_val[0]
        return ret_val

    finally:
        for name, tstate in memento.inp_names + memento.out_names:  # type: ignore[operator]  # concatenation
            memento.sdfg.arrays[name].transient = tstate
