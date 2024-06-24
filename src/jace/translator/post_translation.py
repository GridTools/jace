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
    from dace.sdfg import nodes as dace_nodes

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
    in `trans_ctx.out_names`, will be written into new SDFG variables. In case the
    output variable is a scalar, the output will be replaced by an array of length one.
    This behaviour is consistent with JAX.

    Args:
        trans_ctx: The translation context to process.
    """
    assert trans_ctx.input_names is not None and trans_ctx.out_names is not None

    # NOTE: Currently we do not support to write back into an input argument, as JAX.
    #  However, this is a requirement for handling ICON stencils, that we will support
    #  eventually. If we get a translation context that lists a variable name in the
    #  inputs and outputs, this means that it was returned unmodified. In JAX this
    #  will lead to a copy and we also do it. This is implemented by just naïvely
    #  creating a separate output variable for every output we have, irrespectively
    #  of its name inside the Jaxpr.

    output_pattern = "__jace_output_{}"
    sdfg = trans_ctx.sdfg
    new_output_state: dace.SDFGState = sdfg.add_state("output_processing_stage")
    new_output_names: list[str] = []

    for i, org_output_name in enumerate(trans_ctx.out_names):
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
    trans_ctx.out_names = tuple(new_output_names)


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
    assert trans_ctx.input_names is not None and trans_ctx.out_names is not None

    # NOTE: This function will create a distinct variable for every input. Once we
    #  allow write back arguments they will be handled in the `_create_output_state()`
    #  function anyway, also see the comment in that function.

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
    Finalizes the supplied translation context `trans_ctx`.

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
    if trans_ctx.out_names is None:
        raise ValueError("Output names are not specified.")
    if not (trans_ctx.out_names or trans_ctx.input_names):
        raise ValueError("No input nor output.")

    # We guarantee decoupling
    tsdfg = tjsdfg.TranslatedJaxprSDFG(
        sdfg=copy.deepcopy(trans_ctx.sdfg),
        input_names=trans_ctx.input_names,
        out_names=trans_ctx.out_names,
    )

    # Make inputs and outputs to globals.
    sdfg_arg_names: list[str] = []
    for arg_name in tsdfg.input_names + tsdfg.out_names:
        if arg_name in sdfg_arg_names:
            continue
        tsdfg.sdfg.arrays[arg_name].transient = False
        sdfg_arg_names.append(arg_name)
    tsdfg.sdfg.arg_names = sdfg_arg_names

    if validate:
        tsdfg.validate()
    return tsdfg


def add_nested_sdfg(
    state: dace.SDFGState,
    child_ctx: translator.TranslationContext,
    parent_ctx: translator.TranslationContext,
    in_var_names: Sequence[str],
    out_var_names: Sequence[str],
) -> dace_nodes.NestedSDFG:
    """
    Adds the SDFG in `child_ctx` as nested SDFG at state `state` in `parent_ctx`.

    The function is a convenience wrapper that operates directly on translation
    contexts instead of SDFGs. The function will also create the necessary Memlet
    connections.

    Args:
        state: The state at which the nested SDFG should be inserted.
            Must be part of `parent_ctx`.
        child_ctx: The translation context representing the SDFG that should be added.
        parent_ctx: The parent SDFG to which `child_ctx` should be added as nested
            SDFG in state `state`.
        in_var_names: Names of the variables in `parent_ctx` that are used as inputs for
            the nested SDFG, must have the same order as `child_ctx.input_names`.
        out_var_names: Names of the variables in `parent_ctx` that are used as outputs
            for the nested SDFG, must have the same order as `child_ctx.out_names`.

    Returns:
        The nested SDFG object.

    Note:
        The function will not add `child_ctx` directly as nested SDFG. Instead it
        will first pass it to `finalize_translation_context()` and operates on the
        return values. This means that `child_ctx` will be modified in place, and
        a copy will be added to `parent_ctx`.
        It is highly recommended that `state` is empty.
    """
    if child_ctx.sdfg.free_symbols:
        raise NotImplementedError("Symbol Mapping is not implemented.")
    assert not (child_ctx.input_names is None or child_ctx.out_names is None)  # Silence mypy
    assert len(child_ctx.input_names) == len(in_var_names)
    assert len(child_ctx.out_names) == len(out_var_names)
    assert state in parent_ctx.sdfg.nodes()
    assert not set(in_var_names).intersection(out_var_names)

    if any(input_name.startswith("__jace_mutable_") for input_name in in_var_names):
        raise NotImplementedError(
            "'__jace_mutable_' variables are not yet handled in 'add_nested_sdfg()'."
        )
    if len(set(in_var_names)) != len(in_var_names):
        raise ValueError(
            f"An input can only be passed once, but { {in_var_name for in_var_name in in_var_names if in_var_names.count(in_var_name) > 1} } were passed multiple times."
        )
    if len(set(out_var_names)) != len(out_var_names):
        raise NotImplementedError(
            f"Tried to write multiple times to variables: { {out_var_name for out_var_name in out_var_names if out_var_names.count(out_var_name) > 1} }."
        )

    final_child_ctx = finalize_translation_context(child_ctx)
    nested_sdfg: dace_nodes.NestedSDFG = state.add_nested_sdfg(
        sdfg=final_child_ctx.sdfg,
        parent=parent_ctx.sdfg,
        # Bug in DaCe must be a set.
        inputs=set(final_child_ctx.input_names),
        outputs=set(final_child_ctx.out_names),
    )

    # Now create the connections for the input.
    for outer_name, inner_name in zip(in_var_names, final_child_ctx.input_names):
        outer_array = parent_ctx.sdfg.arrays[outer_name]
        state.add_edge(
            state.add_read(outer_name),
            None,
            nested_sdfg,
            inner_name,
            dace.Memlet.from_array(outer_name, outer_array),
        )

    # Now we create the output connections.
    for outer_name, inner_name in zip(out_var_names, final_child_ctx.out_names):
        outer_array = parent_ctx.sdfg.arrays[outer_name]
        state.add_edge(
            nested_sdfg,
            inner_name,
            state.add_write(outer_name),
            None,
            dace.Memlet.from_array(outer_name, outer_array),
        )

    return nested_sdfg
