# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains all functions that are related to post processing the SDFG.

Most of them operate on `TranslatedJaxprSDFG` objects.

Currently they mostly exist for the sake of existing.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from jace import translator


def postprocess_jaxpr_sdfg(
    trans_ctx: translator.TranslationContext,
    fun: Callable,  # noqa: ARG001  # Currently unused
) -> translator.TranslatedJaxprSDFG:
    """Perform the final post processing steps on the `TranslationContext`.

    Returns:
        The function returns a valid `TranslationContext` that is decoupled from the one
        that was originally part of `trans_ctx`.

    Args:
        trans_ctx:  The `TranslationContext` obtained from the `translate_jaxpr()` function.
        fun:        The original function that was translated.

    Todo:
        - Setting correct input names (layer that does not depend on JAX).
        - Setting the correct strides & Storage properties.
    """
    # Currently we do nothing except finalizing.
    trans_ctx.validate()
    tsdfg: translator.TranslatedJaxprSDFG = copy.deepcopy(trans_ctx.jsdfg)

    finalize_jaxpr_sdfg(tsdfg)

    tsdfg.validate()
    return tsdfg


def finalize_jaxpr_sdfg(
    tsdfg: translator.TranslatedJaxprSDFG,
) -> None:
    """Finalizes the supplied `tsdfg` object in place.

    This function will turn a non finalized, i.e. canonical, SDFG into a finalized one,
    The function will:
    - mark all input and output variables, i.e. listed in `tsdfg.{inp, out}_names`, as globals,
    - set the `arg_names` property of the SDFG,
    """
    if not tsdfg.inp_names:
        raise ValueError("Input names are not specified.")
    if not tsdfg.out_names:
        raise ValueError("Output names are not specified.")

    # Canonical SDFGs do not have global memory, so we must transform it
    sdfg_arg_names: list[str] = []
    for glob_name in tsdfg.inp_names + tsdfg.out_names:
        if glob_name in sdfg_arg_names:  # Donated arguments
            continue
        tsdfg.sdfg.arrays[glob_name].transient = False
        sdfg_arg_names.append(glob_name)

    # This forces the signature of the SDFG to include all arguments in order they appear.
    #  If an argument is used as input and output then it is only listed as input.
    tsdfg.sdfg.arg_names = sdfg_arg_names
