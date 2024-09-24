# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator related handling nested Jaxpr operations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from jax._src import sharding_impls as jax_sharding  # noqa: PLC2701 [import-private-name]

from jace import translator
from jace.translator import post_translation as ptranslation


if TYPE_CHECKING:
    import dace
    from jax._src import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("pjit")
def pjit_translator(
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `pjit` translator that handles nested Jaxpr.

    `pjit` primitives in JAX represents nested calls, for example the branches of a
    conditional are nested Jaxpr. However, in JAX `pjit` is also used to indicate that
    a computation should be done on the device or on sharded memory.
    In case an input is a literal the translator will create a constant for it.

    Args:
        builder: The builder object of the translation.
        in_var_names: Names of the SDFG variables that should be used as inputs
            inside the parent SDFG.
        out_var_names: Names of SDFG variables that should be used as outputs
            inside the parent SDFG.
        eqn: The equation that contains the `pjit` primitive.
        eqn_state: State into which the nested SDFG should be constructed.

    Note:
        The translator ignores the `donated_invars`, the `keep_unused` and the
        `inline` parameter and let's DaCe handle it.
    """
    nested_jaxpr: jax_core.ClosedJaxpr = eqn.params["jaxpr"]
    in_shardings = eqn.params["in_shardings"]
    out_shardings = eqn.params["out_shardings"]
    _ = eqn.params["donated_invars"]  # Always ignored
    _ = eqn.params["keep_unused"]
    _ = eqn.params["inline"]

    if not all(in_sharding is jax_sharding.UNSPECIFIED for in_sharding in in_shardings):
        raise NotImplementedError("Currently 'pjit' does not support sharding in its input.")
    if not all(out_sharding is jax_sharding.UNSPECIFIED for out_sharding in out_shardings):
        raise NotImplementedError("Currently 'pjit' does not support sharding in its output.")

    # TODO(phimuell): Controlflow region and name
    pjit_name = eqn.params["name"]

    # Name in SDFG must be unique, thus we mangle it, furthermore, we have to clean it.
    sdfg_name = f"pjit_{re.subn('[^a-zA-Z0-9_]', '_', pjit_name)[0]}__{'_'.join(out_var_names)}"

    # Ensure that all inputs are SDFG variables
    final_input_names = ptranslation.promote_literals_to_constants(
        builder=builder,
        var_names=in_var_names,
        jax_vars=eqn.invars,
        name_pattern=sdfg_name,
    )

    # Translate the nested expression
    nested_context: translator.TranslationContext = builder.translate_jaxpr(
        jaxpr=nested_jaxpr,
        name=sdfg_name,
    )

    # Now lets add the nested SDFG
    ptranslation.add_nested_sdfg(
        state=eqn_state,
        child_ctx=nested_context,
        parent_ctx=builder._ctx,
        in_var_names=final_input_names,
        out_var_names=out_var_names,
    )
