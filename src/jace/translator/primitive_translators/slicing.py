# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements slicing."""

from __future__ import annotations

from collections.abc import Sequence

import dace
from jax import core as jax_core
from typing_extensions import override

from jace import translator
from jace.translator import mapped_operation_base_translator as mapped_base


class SlicingTranslator(mapped_base.MappedOperationTranslatorBase):
    """Implements the classical slicing operation.

    It is basically a copy Tasklet that only copies parts of the input.
    Note that there is also `dynamic_slice`.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(primitive_name="slice")

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
        """We have to add the offsets to the Memlet accesses."""
        if eqn.params["strides"] is not None:
            raise NotImplementedError("Non 1 strides are not implemented.")

        start_indices = eqn.params["start_indices"]  # Fist index to slice
        return {
            "__in0": dace.Memlet.simple(
                in_var_names[0],
                ", ".join(
                    f"{it_idx} + {start_index}"
                    for (it_idx, _), start_index in zip(tskl_ranges, start_indices)
                ),
            )
        }


translator.register_primitive_translator(SlicingTranslator())
