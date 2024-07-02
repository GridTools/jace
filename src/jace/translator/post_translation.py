# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions for the pre and post processing during the translation."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import dace

from jace import translated_jaxpr_sdfg as tjsdfg, util


if TYPE_CHECKING:
    from jace import translator


def postprocess_jaxpr_sdfg(
    trans_ctx: translator.TranslationContext,
    fun: Callable,  # noqa: ARG001 [unused-function-argument]  # Currently unused.
    flat_call_args: Sequence[Any],
    validate: bool = True,
) -> tjsdfg.TranslatedJaxprSDFG:
    """
    Final post processing steps on the `TranslationContext`.

    While the function performs the post processing on the context in place, the
    returned `TranslatedJaxprSDFG` will be decoupled from the input.

    Args:
        trans_ctx: The `TranslationContext` obtained from a `translate_jaxpr()` call.
        fun: The original function that was translated.
        flat_call_args: The flattened input arguments.
        validate: Perform validation.

    Todo:
        - Fixing the scalar input problem on GPU.
        - Fixing stride problem of the input.
        - Make it such that the context is not modified as a side effect.
    """
    trans_ctx.validate()  # Always validate, it is cheap.
    create_input_output_stages(trans_ctx=trans_ctx, flat_call_args=flat_call_args)
    return finalize_translation_context(trans_ctx, validate=validate)


def create_input_output_stages(
    trans_ctx: translator.TranslationContext, flat_call_args: Sequence[Any]
) -> None:
    """
    Creates an input and output state inside the SDFG in place.

    See `_create_input_state()` and `_create_output_state()` for more information.

    Args:
        trans_ctx: The translation context that should be modified.
        flat_call_args: The flattened call arguments that should be used.

    Note:
        The processed SDFG will remain canonical.
    """
    _create_input_state(trans_ctx, flat_call_args)
    _create_output_state(trans_ctx)


def _create_output_state(trans_ctx: translator.TranslationContext) -> None:
    """
    Creates the output processing stage for the SDFG in place.

    The function will create a new terminal state, in which all outputs, denoted
    in `trans_ctx.output_names`, will be written into new SDFG variables. In case the
    output variable is a scalar, the output will be replaced by an array of length one.
    This behaviour is consistent with JAX.

    Args:
        trans_ctx: The translation context to process.
    """
    assert trans_ctx.input_names is not None and trans_ctx.output_names is not None

    output_pattern = "__jace_output_{}"
    sdfg = trans_ctx.sdfg
    new_output_state: dace.SDFGState = sdfg.add_state("output_processing_stage")
    new_output_names: list[str] = []

    for i, org_output_name in enumerate(trans_ctx.output_names):
        new_output_name = output_pattern.format(i)
        org_output_desc: dace.data.Data = sdfg.arrays[org_output_name]
        assert org_output_desc.transient
        assert (
            new_output_name not in sdfg.arrays
        ), f"Final output variable '{new_output_name}' is already present."

        if isinstance(org_output_desc, dace.data.Scalar):
            _, new_output_desc = sdfg.add_array(
                new_output_name,
                dtype=org_output_desc.dtype,
                shape=(1,),
                transient=True,  # Needed for an canonical SDFG
            )
            memlet = dace.Memlet.simple(new_output_name, subset_str="0", other_subset_str="0")

        else:
            new_output_desc = org_output_desc.clone()
            sdfg.add_datadesc(new_output_name, new_output_desc)
            memlet = dace.Memlet.from_array(org_output_name, org_output_desc)

        new_output_state.add_nedge(
            new_output_state.add_read(org_output_name),
            new_output_state.add_write(new_output_name),
            memlet,
        )
        new_output_names.append(new_output_name)

    sdfg.add_edge(trans_ctx.terminal_state, new_output_state, dace.InterstateEdge())
    trans_ctx.terminal_state = new_output_state
    trans_ctx.output_names = tuple(new_output_names)


def _create_input_state(
    trans_ctx: translator.TranslationContext, flat_call_args: Sequence[Any]
) -> None:
    """
    Creates the input processing state for the SDFG in place.

    The function will create a new set of variables that are exposed as inputs. This
    variables are based on the example input arguments passed through `flat_call_args`.
    This process will hard code the memory location and strides into the SDFG.
    The assignment is performed inside a new state, which is put at the beginning.

    Args:
        trans_ctx: The translation context that should be modified.
        flat_call_args: The flattened call arguments for which the input
            state should be specialized.

    Todo:
        Handle transfer of scalar input in GPU mode.
    """
    assert trans_ctx.input_names is not None and trans_ctx.output_names is not None

    if len(flat_call_args) != len(trans_ctx.input_names):
        raise ValueError(f"Expected {len(trans_ctx.input_names)}, but got {len(flat_call_args)}.")

    sdfg = trans_ctx.sdfg
    new_input_state: dace.SDFGState = sdfg.add_state(f"{sdfg.name}__start_state")
    new_input_names: list[str] = []
    input_pattern = "__jace_input_{}"

    for i, (org_input_name, call_arg) in enumerate(zip(trans_ctx.input_names, flat_call_args)):
        org_input_desc: dace.data.Data = sdfg.arrays[org_input_name]
        new_input_name = input_pattern.format(i)

        if isinstance(org_input_desc, dace.data.Scalar):
            # TODO(phimuell): In GPU mode: scalar -> GPU_ARRAY -> Old input name
            new_input_desc: dace.data.Scalar = org_input_desc.clone()
            sdfg.add_datadesc(new_input_name, new_input_desc)
            memlet = dace.Memlet.simple(new_input_name, subset_str="0", other_subset_str="0")

        else:
            _, new_input_desc = sdfg.add_array(
                name=new_input_name,
                shape=org_input_desc.shape,
                dtype=org_input_desc.dtype,
                strides=util.get_strides_for_dace(call_arg),
                transient=True,  # For canonical SDFG.
                storage=(
                    dace.StorageType.GPU_Global
                    if util.is_on_device(call_arg)
                    else dace.StorageType.CPU_Heap
                ),
            )
            memlet = dace.Memlet.from_array(new_input_name, new_input_desc)

        new_input_state.add_nedge(
            new_input_state.add_read(new_input_name),
            new_input_state.add_write(org_input_name),
            memlet,
        )
        new_input_names.append(new_input_name)

    sdfg.add_edge(new_input_state, trans_ctx.start_state, dace.InterstateEdge())
    sdfg.start_block = sdfg.node_id(new_input_state)
    trans_ctx.start_state = new_input_state
    trans_ctx.input_names = tuple(new_input_names)


def finalize_translation_context(
    trans_ctx: translator.TranslationContext,
    validate: bool = True,
) -> tjsdfg.TranslatedJaxprSDFG:
    """
    Finalizes the translation context and returns a `TranslatedJaxprSDFG` object.

    The function will process the SDFG that is encapsulated inside the context, i.e. a
    canonical one, into a proper SDFG, as it is described in `TranslatedJaxprSDFG`. It
    is important to realize that this function does not perform any optimization of the
    underlying SDFG itself, instead it prepares an SDFG such that it can be passed to
    the optimization pipeline.

    The returned object is fully decoupled from its input and `trans_ctx` is not
    modified.

    Args:
        trans_ctx: The context that should be finalized.
        validate: Call the validate function after the finalizing.
    """
    trans_ctx.validate()
    if trans_ctx.input_names is None:
        raise ValueError("Input names are not specified.")
    if trans_ctx.output_names is None:
        raise ValueError("Output names are not specified.")
    if not (trans_ctx.output_names or trans_ctx.input_names):
        raise ValueError("No input nor output.")

    # We guarantee decoupling
    tsdfg = tjsdfg.TranslatedJaxprSDFG(
        sdfg=copy.deepcopy(trans_ctx.sdfg),
        input_names=trans_ctx.input_names,
        output_names=trans_ctx.output_names,
    )

    # Make inputs and outputs to globals.
    sdfg_arg_names: list[str] = []
    for arg_name in tsdfg.input_names + tsdfg.output_names:
        tsdfg.sdfg.arrays[arg_name].transient = False
        sdfg_arg_names.append(arg_name)
    tsdfg.sdfg.arg_names = sdfg_arg_names

    if validate:
        tsdfg.validate()
    return tsdfg
