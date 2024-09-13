# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements all conditions that are supported in JAX."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import dace

from jace import translator, util
from jace.translator import post_translation as ptranslation
from jace.translator.primitive_translators import pjit_translator as pjit


if TYPE_CHECKING:
    from jax._src import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("cond")
def condition_translator(
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> dace.SDFGState:
    """
    Implements the translation of the `cond` primitive, i.e. a scalar if.

    XLA, JAX' backend, supports two versions, one in which the selector, i.e. the
    variable indicating which branch should be executed is an integer or a boolean.

    Args:
        builder: The builder object of the translation.
        in_var_names: The SDFG variables used an input arguments. First is the index,
            the variable that selects the branch, the remaining ones are passed as
            inputs to the branches.
        out_var_names: Names of SDFG variables that should be used as outputs.
        eqn: The equation that should be translated.
        eqn_state: State into which the nested SDFG should be constructed.

    Returns:
        Because of the nature of this primitive, the translator has to construct
        new states and will return the new SDFG state that serves as terminal state.

    Note:
        The implementation assumes that the selector, i.e. the variables indicating
        which branch should be taken is inside its bound.
    """
    if util.get_jax_var_dtype(eqn.invars[0]) is dace.bool_:
        return _cond_primitive_boolean_impl(
            builder=builder,
            in_var_names=in_var_names,
            out_var_names=out_var_names,
            eqn=eqn,
            eqn_state=eqn_state,
        )
    return _cond_primitive_multi_switch_impl(
        builder=builder,
        in_var_names=in_var_names,
        out_var_names=out_var_names,
        eqn=eqn,
        eqn_state=eqn_state,
    )


def _cond_primitive_multi_switch_impl(
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> dace.SDFGState:
    """
    Implements the integer version of the conditional primitive.

    For arguments see `ConditionTranslator`.

    This [version](https://openxla.org/xla/operation_semantics#conditional) is
    essentially a C switch statement without a default branch.
    """
    # To make names in the SDFG unique we use the name of the equation state
    name_pattern = eqn_state.name

    # Promote all inputs to the branches to variables, this are all except the first
    #  which is the selection variable.
    branch_input_variable_names: list[str] = pjit._promote_literals_to_constants(
        builder=builder,
        var_names=in_var_names[1:],
        jax_vars=eqn.invars[1:],
        name_pattern=name_pattern,
    )

    if in_var_names[0] is None:
        # The selection variable is a literal, so we will now pretend it is a symbol.
        #  This also means that we do not need a state transition to promote the
        #  variable to a symbol.
        selection_symbol = str(util.get_jax_literal_value(eqn.invars[0]))
        selection_state = eqn_state

    else:
        # The selection variable is an input.
        #  For the implementation of the condition we need to promote the selection
        #  variable to a symbol, for which we need an interstate edge.
        #  As a side effect it will update the terminal state.
        selection_variable_name = in_var_names[0]
        selection_symbol = f"{selection_variable_name}_symb"

        selection_state = builder.append_new_state(
            label=f"{name_pattern}_fork",
            assignments={selection_symbol: selection_variable_name},
            prev_state=eqn_state,
        )

    # Now iterate through all branches, translate them and integrate them
    #  for each branch we will generate a dedicated state.
    branch_states: list[dace.SDFGState] = []
    for i, branch_jaxpr in enumerate(eqn.params["branches"]):
        branch_pattern = f"{name_pattern}_{{}}_branch_{i}"
        branch_ctx = builder.translate_jaxpr(jaxpr=branch_jaxpr, name=branch_pattern.format("sdfg"))

        # This will update the terminal state only the first time.
        branch_state = builder.append_new_state(
            label=branch_pattern.format("state"),
            condition=f"{selection_symbol} == {i}",
            prev_state=selection_state,
        )

        # Integrating it.
        ptranslation.add_nested_sdfg(
            state=branch_state,
            child_ctx=branch_ctx,
            parent_ctx=builder._ctx,
            in_var_names=branch_input_variable_names,
            out_var_names=out_var_names,
        )
        branch_states.append(branch_state)

    # Now we have to generate a join state that will serve as new terminal state.
    #  We append it to the first branch state, which is the current terminal state.
    assert builder._terminal_sdfg_state is branch_states[0]
    terminal_state = builder.append_new_state(
        label=f"{name_pattern}_join",
        prev_state=branch_states[0],
    )
    for branch_state in branch_states[1:]:
        builder.sdfg.add_edge(
            branch_state,
            terminal_state,
            dace.sdfg.InterstateEdge(),
        )

    # We return it, because otherwise the builder will assume that `eqn_state` was used.
    return terminal_state


def _cond_primitive_boolean_impl(
    builder: translator.JaxprTranslationBuilder,  # noqa: ARG001 [unused-function-argument]
    in_var_names: Sequence[str | None],  # noqa: ARG001 [unused-function-argument]
    out_var_names: Sequence[str],  # noqa: ARG001 [unused-function-argument]
    eqn: jax_core.JaxprEqn,  # noqa: ARG001 [unused-function-argument]
    eqn_state: dace.SDFGState,  # noqa: ARG001 [unused-function-argument]
) -> dace.SDFGState:
    """
    Implements the case the selector of the primitive is a bool.

    XLA explicitly provides this
    [form of the primitive](https://openxla.org/xla/operation_semantics#conditional)
    JAX however, does not seem to use it and instead forwards it to the integer
    implementation.
    JaCe will not implement it and instead generate an error.
    """
    # NOTE: This is mostly to notice if JAX decided to implement that branch.
    raise NotImplementedError("The boolean conditional primitive is not implemented.")
