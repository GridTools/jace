# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements slicing."""

from __future__ import annotations

import itertools
from collections.abc import MutableSequence, Sequence

import dace
from jax import core as jax_core
from typing_extensions import override

from jace import translator, util
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
        return "__out = __in0"

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


class DynamicSlicingTranslator(translator.PrimitiveTranslator):
    """Implements the dynamic slicing translator.

    The [dynamic slicing](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_slice.html)
    performs a slicing of a _fixed_ window, however, the starting indexes are not fix, but are
    variables that can come from the outside.
    For this it uses symbols that, but since it uses the "Dynamic Map Ranges" no additional state
    is needed.

    Unlike the normal slicing primitive, it is not derived from `MappedOperationTranslatorBase`.
    """

    __slots__ = ()

    @property
    def primitive(self) -> str:
        return "dynamic_slice"

    @override
    def __call__(
        self,
        driver: translator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        assert in_var_names[0]
        assert len(in_var_names) == len(eqn.invars[0].aval.shape) + 1

        # First input to the primitive is the array we slice from, the others are

        in_var_name: str = in_var_names[0]
        start_indices: Sequence[str | None] = in_var_names[1:]

        tskl_ranges: list[tuple[str, str]] = [
            (f"__i{dim}", f"0:{N}") for dim, N in enumerate(eqn.outvars[0].aval.shape)
        ]
        tskl_output: dict[str, dace.Memlet] = {
            "__out": dace.Memlet.simple(
                out_var_names[0],
                ", ".join(name for name, _ in tskl_ranges),
            )
        }

        # Maps the symbol that is used inside the Memlet as offset to the variable where it came from.
        dynamic_map_ranges: dict[str, str] = {}
        mem_accesses: list[str] = []

        for i, (it_var, _), start_idx in zip(itertools.count(), tskl_ranges, start_indices):
            if start_idx is None:  # The index is a literal
                mem_access = f"{it_var} + {util.get_jax_literal_value(eqn.invars[i + 1])}"
            else:
                symb_name = f"__jace_dynamic_map_range_{start_idx}"
                mem_access = f"{it_var} + {symb_name}"
                dynamic_map_ranges[symb_name] = start_idx
            mem_accesses.append(mem_access)

        tskl_input: dict[str, dace.Memlet] = {
            "__in": dace.Memlet.simple(in_var_name, ", ".join(mem_accesses))
        }

        # Now generating the mapped Tasklet.
        tskl_name = f"{self.primitive}_{out_var_names[0]}"

        _, map_entry, _ = eqn_state.add_mapped_tasklet(
            name=tskl_name,
            map_ranges=tskl_ranges,
            inputs=tskl_input,
            code="__out = __in",
            outputs=tskl_output,
            external_edges=True,
        )

        # Now we add the dynamic map indexes.
        for symb_name, start_idx_name in dynamic_map_ranges.items():
            eqn_state.add_edge(
                eqn_state.add_read(start_idx_name),
                None,
                map_entry,
                symb_name,
                dace.Memlet.simple(start_idx_name, "0"),  # It is always a scalar
            )
            map_entry.add_in_connector(symb_name)


translator.register_primitive_translator(SlicingTranslator())
translator.register_primitive_translator(DynamicSlicingTranslator())
