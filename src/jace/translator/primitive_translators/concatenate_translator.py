# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator for concatenation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("concatenate")
def concatenate_translator(
    builder: translator.JaxprTranslationBuilder,  # noqa: ARG001  # Required by the interface.
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `concatenate` primitive.

    Each source array is copied by its own map, but all maps write to the same
    access node.
    """
    if any(in_var_name is None for in_var_name in in_var_names):
        raise NotImplementedError("Concatenate: No literal inputs supported.")

    # Access node that is used by all maps.
    output_nodes = {out_var_names[0]: eqn_state.add_write(out_var_names[0])}

    cat_dim = eqn.params["dimension"]
    cat_offset = 0
    for i, in_var_name in enumerate(in_var_names):
        input_shape = util.get_jax_var_shape(eqn.invars[i])

        tskl_range = [(f"__dim{d}", f"0:{dim_size}") for d, dim_size in enumerate(input_shape)]
        tskl_input_access = [it_var for it_var, _ in tskl_range]

        tskl_output_access = tskl_input_access.copy()
        tskl_output_access[cat_dim] = f"{tskl_output_access[cat_dim]} + {cat_offset}"

        eqn_state.add_mapped_tasklet(
            f"_concatenate_{out_var_names[0]}_{in_var_name}",
            map_ranges=tskl_range,
            inputs={"__in": dace.Memlet.simple(in_var_name, ", ".join(tskl_input_access))},
            code="__out = __in",
            outputs={"__out": dace.Memlet.simple(out_var_names[0], ",".join(tskl_output_access))},
            output_nodes=output_nodes,
            external_edges=True,
        )
        cat_offset += input_shape[cat_dim]
