# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains all functions that are related to post processing the SDFG.

Most of them operate on `TranslatedJaxprSDFG` objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import dace

from jace import translator
from jace.util import dace_helper as jdace


def postprocess_jaxpr_sdfg(
    tsdfg: translator.TranslatedJaxprSDFG,
    fun: Callable,  # noqa: ARG001  # Currently unused
) -> FinalizedJaxprSDFG:
    """Perform the final postprocessing step on the SDFG and returns a finalized version.

    The function will not modify the passed `tsdfg` object (`TranslatedJaxprSDFG`).
    The returned object is of type `FinalizedJaxprSDFG` and is decoupled from the input,
    such that there is no feedback.

    Args:
        tsdfg:  The translated SDFG object.
        fun:    The original function that we translated.
    """
    # Currently we do nothing except finalizing.
    return finalize_jaxpr_sdfg(tsdfg)


def finalize_jaxpr_sdfg(
    trans_sdfg: translator.TranslatedJaxprSDFG,
) -> FinalizedJaxprSDFG:
    """Finalizes the supplied `trans_sdfg` object.

    The returned object is guaranteed to be decoupled from the supplied `TranslatedJaxprSDFG`.
    You should use this function after you have performed all necessary postprocessing for which you need the meta data of the translation.
    The returned object is meant as input for jace's optimization pipeline.

    Note:
        For several reasons this function performs a deep copy of the associated SDFG.
            The enter toolchain assumes and relies on this fact.
    """
    # Check if the outputs are defined.
    if trans_sdfg.inp_names is None:
        raise ValueError("Input names are not specified.")
    if trans_sdfg.out_names is None:
        raise ValueError("Output names are not specified.")

    # We do not support the return value mechanism that dace provides us.
    #  The reasons for that are that the return values are always shared and the working with pytrees is not yet understood.
    #  Thus we make the safe choice by passing all as arguments.
    assert not any(
        arrname.startswith("__return")
        for arrname in trans_sdfg.sdfg.arrays.keys()  # noqa: SIM118  # we can not use `in` because we are also interested in `__return_`!
    ), "Only support SDFGs without '__return' members."

    # We perform a deepcopy by serializing it, as deepcopy is known for having some issues.
    sdfg = dace.SDFG.from_json(trans_sdfg.sdfg.to_json())
    inp_names = trans_sdfg.inp_names
    out_names = trans_sdfg.out_names

    # Canonical SDFGs do not have global memory, so we must transform it
    sdfg_arg_names: list[str] = []
    for glob_name in inp_names + out_names:
        if glob_name in sdfg_arg_names:  # Donated arguments
            continue
        sdfg.arrays[glob_name].transient = False
        sdfg_arg_names.append(glob_name)

    # This forces the signature of the SDFG to include all arguments in order they appear.
    #  If an argument is reused (donated) then it is only listed once, the first time it appears
    sdfg.arg_names = sdfg_arg_names

    return FinalizedJaxprSDFG(sdfg=sdfg, inp_names=inp_names, out_names=out_names)


@dataclass(init=True, eq=False, frozen=False)
class FinalizedJaxprSDFG:
    """This is the final stage of the post processing of the translation.

    Instances of these class only contains enough information to run, but all other meta data associated to tarnslation are lost.
    The idea of this class is that they can be feed to the optimization pipeline of Jace.
    The SDFG that is inside `self` my not be optimized, but input and outputs are marked as global and they have a valid `arg_names` property.

    SDFG encapsulated in `TranslatedJaxprSDFG` is in canonical form, which is not usable, finalized SDFGs are always valid.
    They have:
    - All input an output arrays are marked as global.
    - It does not have `__return` values, i.e. all arguments are passed as arguments.
    - Its `arg_names` are set with set `inp_names + out_names`, however,
        arguments that are input and outputs are only listed as inputs.

    Notes:
        The main reason this class exists is, because optimizations are done in the `JaceLowered.compile()` function.
            All DaCe functions in that regards are in place, if we would not copy the SDFG first, then we would have a problem.
            Because these optimization would then have a feedback of the SDFG object which is stored in one way or the other
            inside the `JaceLowered` object, which is wrong because `jaceLoweredObject.compile(no_opti=False)` and
            `jaceLoweredObject.compile(no_opti=True)` will result in different objects but the SDFG is the same one, i.e. the optimized one.
            It also makes sense, to remove all the unnecessary stuff that is part of the `TranslatedJaxprSDFG` but does not serve any purpose
            inside the optimization pipeline.
    """

    sdfg: dace.SDFG
    inp_names: tuple[str, ...]
    out_names: tuple[str, ...]
    csdfg: jdace.CompiledSDFG | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> bool:
        """Checks if the embedded SDFG is valid."""
        if any(arrname.startswith("__return") for arrname in self.sdfg.arrays.keys()):  # noqa: SIM118  # we can not use `in` because we are also interested in `__return_`!
            raise dace.sdfg.InvalidSDFGError(
                "There are no output arguments.",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        for glob_name in self.inp_names + self.out_names:
            if self.sdfg.arrays[glob_name].transient:
                raise dace.sdfg.InvalidSDFGError(
                    f"Argument '{glob_name}' is a transient.",
                    self.sdfg,
                    self.sdfg.node_id(self.sdfg.start_state),
                )
        self.sdfg.validate()
        return True
