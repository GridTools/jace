# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions for the pre and post processing during the translation."""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any

import dace
import jax
from jax import tree_util as jax_tree

import jace
from jace import util


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from jace import translator


def postprocess_jaxpr_sdfg(
    trans_ctx: translator.TranslationContext,
    fun: Callable,  # noqa: ARG001  # Currently unused
    call_args: Sequence[Any],
    validate: bool = True,
) -> jace.TranslatedJaxprSDFG:
    """
    Final post processing steps on the `TranslationContext`.

    While the function performs the post processing on the context in place, the
    returned `TranslatedJaxprSDFG` will be decoupled from the input.

    Args:
        trans_ctx: The `TranslationContext` obtained from a `translate_jaxpr()` call.
        fun: The original function that was translated.
        call_args: The flattened input arguments.
        validate: Perform validation.

    Todo:
        - Fixing the scalar input problem on GPU.
        - Fixing stride problem of the input.
    """
    trans_ctx.validate()  # Always validate, it is cheap.
    create_input_output_stages(trans_ctx=trans_ctx, call_args=call_args)
    return finalize_translation_context(trans_ctx, validate=validate)


def create_input_output_stages(
    trans_ctx: translator.TranslationContext, call_args: Sequence[Any]
) -> None:
    """
    Creates an input and output state inside the SDFG in place.

    See `_create_input_state()` and `_create_output_state()` for more information.

    Args:
        trans_ctx: The translation context that should be modified.
        call_args: The flattened call arguments that should be used.

    Note:
        The processed SDFG will remain canonical.
    """
    _create_input_state(trans_ctx, call_args)
    _create_output_state(trans_ctx)


def _create_output_state(trans_ctx: translator.TranslationContext) -> None:
    """
    Creates the output processing stage for the SDFG in place.

    The function will create a new terminal state, in which all outputs, denoted
    in `trans_ctx.out_names`, will be written into new SDFG variables. In case the
    output variable is a scalar, the output will be replaced by an array of length one.
    This behaviour is consistent with Jax.

    Args:
        trans_ctx: The translation context to process.
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


def _create_input_state(trans_ctx: translator.TranslationContext, call_args: Sequence[Any]) -> None:
    """
    Creates the input processing state for the SDFG in place.

    The function will create a new set of variables that are exposed as inputs. If an
    input argument is an array, the new variable will have the same strides and storage
    location the actual input value, that is passed inside `call_args`. If the input is
    a scalar and GPU mode is activated, the function will add the necessary connections
    to transfer it to the device.

    Args:
        trans_ctx: The translation context that should be modified.
        call_args: The flattened call arguments for which the input
            state should be specialized.

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
    trans_ctx.inp_names = tuple(new_input_names)


def finalize_translation_context(
    trans_ctx: translator.TranslationContext,
    validate: bool = True,
) -> jace.TranslatedJaxprSDFG:
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
    if trans_ctx.inp_names is None:
        raise ValueError("Input names are not specified.")
    if trans_ctx.out_names is None:
        raise ValueError("Output names are not specified.")
    if not (trans_ctx.out_names or trans_ctx.inp_names):
        raise ValueError("No input nor output.")

    # We guarantee decoupling
    tsdfg = jace.TranslatedJaxprSDFG(
        sdfg=copy.deepcopy(trans_ctx.sdfg),
        inp_names=trans_ctx.inp_names,
        out_names=trans_ctx.out_names,
    )

    # Make inputs and outputs to globals.
    sdfg_arg_names: list[str] = []
    for arg_name in tsdfg.inp_names + tsdfg.out_names:
        if arg_name in sdfg_arg_names:
            continue
        tsdfg.sdfg.arrays[arg_name].transient = False
        sdfg_arg_names.append(arg_name)
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
    """
    Traces `fun` and generates the Jaxpr and some related meta data.

    For tracing the computation `fun` the function uses the `trace_call_args`
    and `trace_call_kwargs` arguments, both should not be flattened. Furthermore,
    the tracing is done in enabled x64 mode.

    Returns:
        The function will return a tuple of length three.
        1) The Jaxpr that was generated by Jax using the supplied arguments and options.
        2) The flattened input.
        3) A pytree describing the output.

    Args:
        fun: The original Python computation.
        trace_call_args: The positional arguments that should be used for
            tracing the computation.
        trace_call_kwargs: The keyword arguments that should be used for
            tracing the computation.
        trace_options: The options used for tracing, the same arguments that
            are supported by `jace.jit`.

    Todo:
        - Handle default arguments of `fun`.
        - Handle static arguments.
        - Turn `trace_options` into a `TypedDict` and sync with `jace.jit`.
    """
    if trace_options:
        raise NotImplementedError(
            f"Not supported tracing options: {', '.join(f'{k}' for k in trace_options)}"
        )
    assert all(param.default is param.empty for param in inspect.signature(fun).parameters.values())

    # In Jax `float32` is the main datatype, and they go to great lengths to avoid some
    #  aggressive [type promotion](https://jax.readthedocs.io/en/latest/type_promotion.html).
    #  However, in this case we will have problems when we call the SDFG, for some
    #  reasons `CompiledSDFG` does not work in that case correctly, thus we enable it
    #  for the tracing.
    with jax.experimental.enable_x64():
        # TODO(phimuell): copy the implementation of the real tracing
        jaxpr, outshapes = jax.make_jaxpr(fun, return_shape=True)(
            *trace_call_args, **trace_call_kwargs
        )

    # Regardless what the documentation of `make_jaxpr` claims, it does not output a
    #  pytree instead an abstract description of the shape, that we will transform into
    #  a pytree.
    outtree = jax_tree.tree_structure(outshapes)

    # Make the input tree
    flat_in_vals = jax_tree.tree_leaves((trace_call_args, trace_call_kwargs))
    assert len(jaxpr.in_avals) == len(flat_in_vals), "Static arguments not implemented."

    return jaxpr, flat_in_vals, outtree
