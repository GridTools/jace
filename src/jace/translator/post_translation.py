# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains all functions that are related to post processing the SDFG.

Most of them operate on `TranslatedJaxprSDFG` objects.
Currently they mostly exist for the sake of existing.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from jace import translator


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def postprocess_jaxpr_sdfg(
    trans_ctx: translator.TranslationContext,
    fun: Callable,  # noqa: ARG001  # Currently unused
    call_args: Sequence[Any],  # noqa: ARG001  # Currently unused
    intree: None,  # noqa: ARG001  # Currently unused
) -> translator.TranslatedJaxprSDFG:
    """
    Perform the final post processing steps on the `TranslationContext` _in place_.

    The function will perform post processing stages on the context in place.
    However, the function will return a decoupled `TranslatedJaxprSDFG` object.

    Args:
        trans_ctx: The `TranslationContext` obtained from a `translate_jaxpr()` call.
        fun: The original function that was translated.
        call_args: The linearized input arguments.
        intree: The pytree describing the inputs.

    Todo:
        - Setting correct input names (layer that does not depend on JAX).
        - Setting the correct strides & storage properties.
        - Fixing the scalar input problem on GPU.
    """
    # Currently we do nothing except finalizing.
    trans_ctx.validate()

    #
    # Assume some post processing here.
    #

    return finalize_translation_context(trans_ctx, validate=True)


def finalize_translation_context(
    trans_ctx: translator.TranslationContext, validate: bool = True
) -> translator.TranslatedJaxprSDFG:
    """
    Finalizes the supplied translation context `trans_ctx`.

    The function will process the SDFG that is encapsulated inside the context,
    i.e. a canonical one, into a proper SDFG, as it is described in
    `TranslatedJaxprSDFG`. It is important to realize that this function does
    not perform any optimization of the underlying SDFG itself, instead it
    prepares an SDFG such that it can be passed to the optimization pipeline.

    The function will not mutate the passed translation context and the output
    is always decoupled from its output.

    Args:
        trans_ctx: The context that should be finalized.
        validate: Call the validate function after the finalizing.
    """
    trans_ctx.validate()
    if trans_ctx.inp_names is None:
        raise ValueError("Input names are not specified.")
    if trans_ctx.out_names is None:
        raise ValueError("Output names are not specified.")

    # We guarantee decoupling
    tsdfg = translator.TranslatedJaxprSDFG(
        sdfg=copy.deepcopy(trans_ctx.sdfg),
        inp_names=trans_ctx.inp_names,
        out_names=trans_ctx.out_names,
    )

    # Make inputs and outputs to globals.
    sdfg_arg_names: list[str] = []
    for glob_name in tsdfg.inp_names + tsdfg.out_names:
        if glob_name in sdfg_arg_names:
            continue
        tsdfg.sdfg.arrays[glob_name].transient = False
        sdfg_arg_names.append(glob_name)

    # This forces the signature of the SDFG to include all arguments in order they
    #  appear. If an argument is used as input and output then it is only listed as
    #  input.
    tsdfg.sdfg.arg_names = sdfg_arg_names

    if validate:
        tsdfg.validate()

    return tsdfg
