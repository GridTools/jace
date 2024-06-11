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
import inspect
from typing import TYPE_CHECKING, Any

import dace
import jax
from jax import tree_util as jax_tree

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def postprocess_jaxpr_sdfg(
    trans_ctx: translator.TranslationContext,
    fun: Callable,  # noqa: ARG001  # Currently unused
    call_args: Sequence[Any],
    outtree: jax_tree.PyTreeDef,
) -> translator.TranslatedJaxprSDFG:
    """Perform the final post processing steps on the `TranslationContext` _in place_.

    The function will perform post processing stages on the context in place.
    However, the function will return a decoupled `TranslatedJaxprSDFG` object.

    Args:
        trans_ctx: The `TranslationContext` obtained from the `translate_jaxpr()` function.
        fun: The original function that was translated.
        call_args: The flattened input arguments.
        outtree: A pytree describing how to unflatten the output.

    Todo:
        - Fixing the scalar input problem on GPU.
        - Fixing stride problem of the input.
    """
    # Currently we do nothing except finalizing.
    trans_ctx.validate()

    # Handle inputs
    create_input_output_stages(trans_ctx=trans_ctx, call_args=call_args)

    return finalize_translation_context(trans_ctx, outtree=outtree, validate=True)


def create_input_output_stages(
    trans_ctx: translator.TranslationContext, call_args: Sequence[Any]
) -> None:
    """Creates an input and output state inside the SDFG in place.

    Args:
        trans_ctx: The translation context that should be modified.
        call_args: the call arguments that should be used.
    """
    _create_input_state(trans_ctx, call_args)
    _create_output_state(trans_ctx)


def _create_output_state(trans_ctx: translator.TranslationContext) -> None:
    """Creates the output processing stage for the SDFG in place.

    The function will create a new terminal state, in which all outputs, denoted
    in `trans_ctx.out_names` will be written in new SDFG variables. However,
    instead of scalars the function will generate arrays of length one. This is
    needed because DaCe can only return arrays at the moment, it is also
    consistent with what Jax does.

    Notes:
        All output variables follow the pattern `__jace_output_{i}`, where `i`
        is a zero based counter. Furthermore, all output variables are transients
        since `TranslationContext` is supposed to hold canonical SDFGs only.
    """
    assert trans_ctx.inp_names is not None and trans_ctx.out_names is not None

    if set(trans_ctx.inp_names).intersection(trans_ctx.out_names):
        raise NotImplementedError("Shared input and output variables are not supported yet.")

    output_pattern = "__jace_output_{}"
    sdfg = trans_ctx.sdfg
    new_output_state: dace.SDFGState = sdfg.add_state("output_processing_stage")
    new_output_names: list[str] = []

    for i, org_output_name in enumerate(trans_ctx.out_names):
        new_output_name = output_pattern.format(i)
        org_output_desc: dace.data.Data = sdfg.arrays[org_output_name]

        if isinstance(org_output_desc, dace.data.Scalar):
            _, new_output_desc = sdfg.add_array(
                new_output_name,
                dtype=org_output_desc.dtype,
                shape=(1,),
                transient=True,
                strides=None,  # explicit C stride
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


def _create_input_state(trans_ctx: translator.TranslationContext, call_args: Sequence[Any]) -> None:
    """Creates the input processing state for the SDFG in place.

    The function creates a new set of variables that are exposed as inputs, whose
    names follows the pattern `__jace_input_{i}`, where `i` is a zero based
    counter. These new variables will have the same strides as the input array.
    Furthermore, they will have the correct storage locations and scalars in
    GPU mode will be handled correctly.

    Args:
        trans_ctx: The translation context that should be modified.
        call_args: the call arguments that should be used.

    Todo:
        Handle transfer of scalar input in GPU mode.
    """
    assert trans_ctx.inp_names is not None and trans_ctx.out_names is not None

    if set(trans_ctx.inp_names).intersection(trans_ctx.out_names):
        raise NotImplementedError("Shared input and output variables are not supported yet.")
    if len(call_args) != len(trans_ctx.inp_names):
        raise ValueError(f"Expected {len(trans_ctx.inp_names)}, but got {len(call_args)}.")

    sdfg = trans_ctx.sdfg
    new_input_state: dace.SDFGState = sdfg.add_state(f"{sdfg.name}__start_state")
    new_input_names: list[str] = []
    input_pattern = "__jace_input_{}"

    for i, (org_input_name, call_arg) in enumerate(zip(trans_ctx.inp_names, call_args)):
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
                transient=True,
                storage=dace.StorageType.GPU_Global
                if util.is_on_device(call_arg)
                else dace.StorageType.CPU_Heap,
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
    trans_ctx.inp_names = tuple(new_input_names)


def finalize_translation_context(
    trans_ctx: translator.TranslationContext, outtree: jax_tree.PyTreeDef, validate: bool = True
) -> translator.TranslatedJaxprSDFG:
    """Finalizes the supplied translation context `trans_ctx`.

    The function will process the SDFG that is encapsulated inside the context,
    i.e. a canonical one, into a proper SDFG, as it is described in
    `TranslatedJaxprSDFG`. It is important to realize that this function does
    not perform any optimization of the underlying SDFG itself, instead it
    prepares an SDFG such that it can be passed to the optimization pipeline.

    The function will not mutate the passed translation context and the output
    is always decoupled from its output.

    Args:
        trans_ctx: The context that should be finalized.
        outtree: A pytree describing how to restore the output.
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
        outtree=outtree,
    )

    # Make inputs and outputs to globals.
    sdfg_arg_names: list[str] = []
    for arg_name in tsdfg.inp_names + tsdfg.out_names:
        if arg_name in sdfg_arg_names:
            continue
        tsdfg.sdfg.arrays[arg_name].transient = False
        sdfg_arg_names.append(arg_name)

    # This forces the signature of the SDFG to include all arguments in order they appear.
    #  If an argument is used as input and output then it is only listed as input.
    tsdfg.sdfg.arg_names = sdfg_arg_names

    if validate:
        tsdfg.validate()

    return tsdfg


def trace_and_flatten_function(
    fun: Callable,
    trace_call_args: Sequence[Any],
    trace_call_kwargs: Mapping[str, Any],
    trace_options: Mapping[str, Any],
) -> tuple[jax.core.ClosedJaxpr, list[Any], jax_tree.PyTreeDef]:
    """Traces `fun` and generates the Jaxpr as well as the input and output tree.

    The function will perform the tracing using `trace_options`, which are the
    same as supported by `jace.jit`. Furthermore the tracing is done with
    x64 enabled.

    Returns:
        The function will return a tuple of length three.
        1) The Jaxpr that was generated by tracing using the supplied arguments
            and options.
        2) The flattened input values.
        3) A pytree describing the output structure.

    Todo:
        - Handle default arguments of `fun`.
        - Handle static arguments.
    """
    if trace_options:
        raise NotImplementedError(
            f"Not supported tracing options: {', '.join(f'{k}' for k in trace_options)}"
        )
    assert all(param.default is param.empty for param in inspect.signature(fun).parameters.values())

    # In Jax `float32` is the main datatype, and they go to great lengths to avoid some
    #  aggressive [type promotion](https://jax.readthedocs.io/en/latest/type_promotion.html).
    #  However, in this case we will have problems when we call the SDFG, for some reasons
    #  `CompiledSDFG` does not work in that case correctly, thus we enable it for the tracing.
    with jax.experimental.enable_x64():
        # TODO(phimuell): copy the implementation of the real tracing, and not the debug one.
        jaxpr, outshapes = jax.make_jaxpr(fun, return_shape=True)(
            *trace_call_args, **trace_call_kwargs
        )

    # Regardless what the documentation of `make_jaxpr` claims, it does not output a pytree.
    #  instead an abstract description of the shape, that we will transform into a pytree.
    outtree = jax_tree.tree_structure(outshapes)

    # Make the input tree
    flat_in_vals = jax_tree.tree_leaves((trace_call_args, trace_call_kwargs))
    assert len(jaxpr.in_avals) == len(flat_in_vals), "Static arguments not implemented."

    return jaxpr, flat_in_vals, outtree
