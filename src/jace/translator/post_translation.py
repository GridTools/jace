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

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from jace import translator


def postprocess_jaxpr_sdfg(
    tsdfg: translator.TranslatedJaxprSDFG,
    fun: Callable,  # noqa: ARG001  # Currently unused
) -> None:
    """Perform the final post processing steps on the SDFG in place.

    Afterwards `tsdfg` will be finalized.

    Args:
        tsdfg:  The translated SDFG object.
        fun:    The original function that we translated.

    Todo:
        - Setting correct input names (layer that does not depend on JAX).
        - Setting the correct strides & Storage properties.
    """
    # Currently we do nothing except finalizing.
    finalize_jaxpr_sdfg(tsdfg)


def finalize_jaxpr_sdfg(
    tsdfg: translator.TranslatedJaxprSDFG,
) -> None:
    """Finalizes the supplied `tsdfg` object in place.

    This function will turn a non finalized, i.e. canonical, SDFG into a finalized one,
    i.e. after this function `tsdfg.is_finalized` is `True`.
    The function will:
    - mark all input and output variables, i.e. listed in `tsdfg.{inp, out}_names`, as globals,
    - set the `arg_names` property of the SDFG,
    - deallocate all members of `tsdfg` that are no longer needed.
    """
    if tsdfg.is_finalized:
        raise ValueError("The supplied SDFG is already finalized.")
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

    # Now we will deallocate the fields and mark `self` as finalized.
    tsdfg.start_state = None
    tsdfg.terminal_state = None
    tsdfg.is_finalized = True
