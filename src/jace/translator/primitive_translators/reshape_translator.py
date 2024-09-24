# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator for reshaping operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("reshape")
def reshape_translator(
    builder: translator.JaxprTranslationBuilder,  # noqa: ARG001 [unused-function-argument]  # Required by the interface.
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `reshape` primitive.

    The function creates a memlet between the input (old shape) and output (final
    shape). Because of this, it is best if both arrays do not have any paddings.

    Args:
        builder: The builder object of the translation.
        in_var_names: Name of the SDFG variable of the source array,
            with the old shape.
        out_var_names: Name of SDFG variable that acts as destination,
            with the new shape.
        eqn: The equation that contains the `pjit` primitive.
        eqn_state: State into which the nested SDFG should be constructed.

    Note:
        The optional `dimensions` parameters, which allows to permute the input,
        is not supported.
    """
    if eqn.params["dimensions"] is not None:
        raise NotImplementedError("Currently 'dimensions' must be 'None'.")
    eqn_state.add_nedge(
        eqn_state.add_read(in_var_names[0]),
        eqn_state.add_write(out_var_names[0]),
        dace.Memlet(
            data=in_var_names[0],
            subset=", ".join(f"0:{size}" for size in util.get_jax_var_shape(eqn.invars[0])),
            other_subset=", ".join(f"0:{size}" for size in eqn.params["new_sizes"]),
        ),
    )
