# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator for condition operations, i.e. scalar `if` and `switch`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import dace

from jace import translator, util
from jace.translator import post_translation as ptranslation


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
    Implements the translation of scalar conditional branches.

    This translator handles both `jax.lax.cond()` and `jax.lax.switch()` cases.
    The sub expression of the branches are each translated into a separate nested
    SDFG, each located in their own state. These state are then connected to the
    joint state which is returned.

    Args:
        builder: The builder object of the translation.
        in_var_names: The SDFG variables used an input arguments. First is the
            selection variable. The remaining ones are passed to the branches as
            inputs.
        out_var_names: Names of SDFG variables that should be used as outputs.
        eqn: The equation that should be translated.
        eqn_state: State into which the nested SDFG should be constructed.

    Notes:
        - According to the JAX documentation (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html)
            the selector is clamped. But according to XLA (https://openxla.org/xla/operation_semantics#conditional)
            an out of range selector uses the last branch. JaCe conforms to JAX
            semantic.
        - After this function the terminal state of the `builder` is unspecific.
    """
    if util.get_jax_var_dtype(eqn.invars[0]) is dace.bool_:
        # XLA explicitly provides a binary form of the primitive
        #  (https://openxla.org/xla/operation_semantics#conditional) JAX however,
        #  does not seem to use it at the moment and instead forwards it to the
        #  integer implementation.
        raise NotImplementedError("The boolean conditional primitive is not implemented.")

    # Used as prefix to give all additional states/variables a unique name.
    name_pattern = eqn_state.name

    # To avoid special cases promote all symbols to constants.
    branch_input_variable_names: list[str] = ptranslation.promote_literals_to_constants(
        builder=builder,
        var_names=in_var_names[1:],
        jax_vars=eqn.invars[1:],
        name_pattern=name_pattern,
    )

    # expressions of the branches.
    branches: list[jax_core.ClosedJaxpr] = eqn.params["branches"]

    # Make sure that the selection variable is a DaCe symbol.
    if in_var_names[0] is None:
        literal_selection_value = str(util.get_jax_literal_value(eqn.invars[0]))
        selection_symbol = f"min({len(branches)}, max(0, {literal_selection_value}))"
        selection_state = eqn_state
    else:
        selection_variable_name = in_var_names[0]
        selection_symbol = f"{selection_variable_name}_symb"
        selection_state = builder.append_new_state(
            label=f"{name_pattern}_fork",
            assignments={
                selection_symbol: f"min({len(branches)}, max(0, {selection_variable_name}))"
            },
            prev_state=eqn_state,
        )

    # Translate the subbranches, the branches are all connected from `selection_state`.
    branch_states: list[dace.SDFGState] = []
    for i, branch_jaxpr in enumerate(branches):
        branch_pattern = f"{name_pattern}_{{}}_branch_{i}"
        branch_ctx = builder.translate_jaxpr(jaxpr=branch_jaxpr, name=branch_pattern.format("sdfg"))

        # The first time it is called it will update the builder's terminal state
        #  but since we will return the join state it will be updated later. But
        #  until then the terminal state of the builder is invalid.
        branch_state = builder.append_new_state(
            label=branch_pattern.format("state"),
            condition=f"{selection_symbol} == {i}",
            prev_state=selection_state,
        )
        ptranslation.add_nested_sdfg(
            state=branch_state,
            child_ctx=branch_ctx,
            parent_ctx=builder._ctx,
            in_var_names=branch_input_variable_names,
            out_var_names=out_var_names,
        )
        branch_states.append(branch_state)

    # Connect all branch states to the join state
    join_state = builder._ctx.sdfg.add_state(label=f"{name_pattern}__join_state")
    for branch_state in branch_states:
        builder.sdfg.add_edge(
            branch_state,
            join_state,
            dace.sdfg.InterstateEdge(),
        )

    return join_state
