# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the `pjit` translator, i.e. nested Jaxpr expressions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from jax._src import sharding_impls as jax_sharding  # noqa: PLC2701 [import-private-name]

from jace import translator
from jace.translator import post_translation as ptranslation


if TYPE_CHECKING:
    import dace
    from jax._src import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("pjit")
def PJITTranslator(  # noqa: N802 [invalid-function-name]
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `pjit` translator that handles nested Jaxpr.

    `pjit` primitives in JAX represents nested calls, for example the body of a scan
    is inside a nested Jaxpr. However, `pjit` is used to indicate that a computation
    should be done on the device or on sharded memory.

    However, due to the current state and working of JaCe, this aspect is essentially
    ignored and the computation is always inlined.

    Args:
        builder: The builder object of the translation.
        in_var_names: Names of the SDFG variables that should be used as inputs
            inside the parent SDFG.
        out_var_names: Names of SDFG variables that should be used as outputs
            inside the parent SDFG.
        eqn: The equation that contains the `pjit` primitive.
        eqn_state: State into which the nested SDFG should be constructed.
    """
    params: dict[str, Any] = eqn.params
    nested_jaxpr: jax_core.ClosedJaxpr = params["jaxpr"]
    in_shardings = params["in_shardings"]
    out_shardings = params["out_shardings"]
    _ = params["donated_invars"]  # Always ignored
    _ = params["keep_unused"]
    _ = params["inline"]

    # TODO(phimuell): Controlflow region and name
    pjit_name = params["name"]

    if not all(in_sharding is jax_sharding.UNSPECIFIED for in_sharding in in_shardings):
        raise NotImplementedError("Currently 'pjit' does not support sharding in its input.")
    if not all(out_sharding is jax_sharding.UNSPECIFIED for out_sharding in out_shardings):
        raise NotImplementedError("Currently 'pjit' does not support sharding in its output.")
    if any(in_var_name is None for in_var_name in in_var_names):
        raise NotImplementedError("Literal inputs to 'pjit' are not implemented.")

    # TODO(phimuell): Controlflow region and name
    #  They will introduce a feature like that to address them in optimizations.
    pjit_name = params["name"]

    # Name in SDFG must be unique, thus we mangle it, furthermore, we have to clean it.
    sdfg_name = f"pjit_{re.subn('[^a-zA-Z0-9_]', '_', pjit_name)[0]}__{'_'.join(out_var_names)}"

    # Now get the translated SDFG.
    nested_context: translator.TranslationContext = builder.translate_jaxpr(
        jaxpr=nested_jaxpr,
        name=sdfg_name,
    )

    # Now lets add the nested SDFG
    ptranslation.add_nested_sdfg(
        state=eqn_state,
        child_ctx=nested_context,
        parent_ctx=builder._ctx,
        in_var_names=in_var_names,  # type: ignore[arg-type]  # Is checked above.
        out_var_names=out_var_names,
    )
