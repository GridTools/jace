# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the concatenation primitive."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace
from typing_extensions import override

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class ConcatenateTranslator(translator.PrimitiveTranslator):
    """
    Implements the `concatenate` primitive.

    It is implemented by a series of map that writes to the same access node.
    It is probably the largest stretch of "written once" in the entire core.
    """

    @property
    def primitive(self) -> str:  # noqa: D102  # No docstring needed.
        return "concatenate"

    @override
    def __call__(
        self,
        builder: translator.JaxprTranslationBuilder,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        if any(in_var_name is None for in_var_name in in_var_names):
            raise NotImplementedError("Concatenate: No literal inputs supported.")

        # Dimension along we concatenate.
        cat_dim = eqn.params["dimension"]

        # Offset counter for write back.
        already_copied = 0

        # This is the access node we use for the output
        #  Is inside a dict for input to `add_mapped_tasklet()`.
        output_nodes = {out_var_names[0]: eqn_state.add_write(out_var_names[0])}

        # Now going over each input and copying the input in the correct location
        #  of the output array.
        for i, in_var_name in enumerate(in_var_names):
            input_shape = util.get_jax_var_shape(eqn.invars[i])

            tskl_range = [(f"__dim{d}", f"0:{dim_size}") for d, dim_size in enumerate(input_shape)]
            tskl_input_access = [it_var for it_var, _ in tskl_range]

            tskl_output_access = tskl_input_access.copy()
            tskl_output_access[cat_dim] = f"{tskl_output_access[cat_dim]} + {already_copied}"

            eqn_state.add_mapped_tasklet(
                f"_concatenate_{out_var_names[0]}_{in_var_name}",
                map_ranges=tskl_range,
                inputs={"__in": dace.Memlet.simple(in_var_name, ", ".join(tskl_input_access))},
                code="__out = __in",
                outputs={
                    "__out": dace.Memlet.simple(out_var_names[0], ",".join(tskl_output_access))
                },
                output_nodes=output_nodes,
                external_edges=True,
            )

            # Update the counter that we have copied
            already_copied += input_shape[cat_dim]


_ = translator.register_primitive_translator(ConcatenateTranslator())
