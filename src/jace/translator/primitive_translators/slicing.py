# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements slicing."""

from __future__ import annotations

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

    Note:
        Jax will adjust the start indexes if the window overrun, however, Jace will not do that.
        Instead, Jace will consider this as undefined behaviour.

    Todo:
        Fix the divergence with Jax, for this pre process the start indexes by the following
        formula $min(s + w, N) - w$, where $s$ is the start index, $w$ the window size and
        $N$ the length in a particular dimension, for this we need Tasklets, if we want to
        preserve merge ability.
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

        # The first input to the primitive is the array we slice from, the others are the start
        #  indices of the slice window, each is a scalar, maybe literals
        in_var_name: str = in_var_names[0]
        start_indices: Sequence[str | None] = in_var_names[1:]

        tskl_ranges: list[tuple[str, str]] = [
            (f"__i{dim}", f"0:{N}") for dim, N in enumerate(eqn.outvars[0].aval.shape)
        ]

        # We use dynamic map ranges, thus the map entry has entries, not with the typical `IN_*`
        #  name and the connector name defines a symbol within the map scope. This `dict` maps
        #  the symbol name to the name of the input variable, that defines the symbol. If the
        #  input is a literal, than it has no correspondence and the constant is substituted.
        dynamic_map_ranges: dict[str, str] = {}
        memlet_accesses: list[str] = []

        for i, ((it_var, _), start_idx) in enumerate(zip(tskl_ranges, start_indices)):
            if start_idx is None:
                offset = str(util.get_jax_literal_value(eqn.invars[i + 1]))
            else:
                offset = f"__jace_dynamic_map_range_{out_var_names[0]}_{start_idx}"
                dynamic_map_ranges[offset] = start_idx
            memlet_accesses.append(f"{it_var} + {offset}")

        tskl_input = dace.Memlet.simple(in_var_name, ", ".join(memlet_accesses))
        tskl_output = dace.Memlet.simple(
            out_var_names[0],
            ", ".join(name for name, _ in tskl_ranges),
        )

        _, map_entry, _ = eqn_state.add_mapped_tasklet(
            name=f"{self.primitive}_{out_var_names[0]}",
            map_ranges=tskl_ranges,
            inputs={"__in": tskl_input},
            code="__out = __in",
            outputs={"__out": tskl_output},
            external_edges=True,
        )

        # Creating the inputs for the dynamic map ranges.
        for symb_name, start_idx_name in dynamic_map_ranges.items():
            eqn_state.add_edge(
                eqn_state.add_read(start_idx_name),
                None,
                map_entry,
                symb_name,
                dace.Memlet.simple(start_idx_name, "0"),
            )
            map_entry.add_in_connector(symb_name)


translator.register_primitive_translator(SlicingTranslator())
translator.register_primitive_translator(DynamicSlicingTranslator())
