# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This implements the `broadcast_in_dim` primitive."""

from __future__ import annotations

from collections.abc import Sequence

import dace
from jax import core as jax_core
from typing_extensions import override

from jace import translator
from jace.translator.primitive_translators.mapped_operation_base_translator import (
    MappedOperationTranslatorBase,
)


class BroadcastInDimTranslator(MappedOperationTranslatorBase):
    """This handles the `broadcast_in_dim` primitives."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(primitive_name="broadcast_in_dim")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        return "__in0"

    @override
    def make_input_memlets(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> dict[str, dace.Memlet]:
        if in_var_names[0] is None:
            return {}
        return {
            "__in0": dace.Memlet.simple(
                in_var_names[0],
                ", ".join(tskl_ranges[bdim][0] for bdim in eqn.params["broadcast_dimensions"])
                if eqn.params["broadcast_dimensions"]
                else "0",
            )
        }


translator.register_primitive_translator(BroadcastInDimTranslator())
