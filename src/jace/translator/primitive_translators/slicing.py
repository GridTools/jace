# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements slicing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace
from typing_extensions import override

from jace import translator, util
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class SlicingTranslator(mapped_base.MappedOperationTranslatorBase):
    """
    Implements the `slice` primitive.

    This is the classical slicing operation which extracts a fixed sized window
    from a fixed initial position. The slicing is implemented using a partial copy.

    Note:
        Slices are essentially optimization barriers as they can not be fused
        with Maps before them.
    """

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
        strides: Sequence[int] = (
            ((1,) * len(tskl_ranges)) if eqn.params["strides"] is None else eqn.params["strides"]
        )
        start_indices: Sequence[int] = eqn.params["start_indices"]  # Fist index to slice
        return {
            "__in0": dace.Memlet.simple(
                in_var_names[0],
                ", ".join(
                    f"{start_index} + {it_idx} * {stride}"
                    for (it_idx, _), start_index, stride in zip(tskl_ranges, start_indices, strides)
                ),
            )
        }


class DynamicSlicingTranslator(translator.PrimitiveTranslator):
    """
    Implements the `dynamic_slice` primitive.

    [Dynamic slicing](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_slice.html)
    performs a slicing of a _fixed_ window, but the start of the window is
    not fix, instead it is passed by variables. Furthermore, (as it is in Jax),
    if the window would overrun the start indexes are adjusted.

    Todo:
        - Prevent that the modified start indexes are promoted to symbols,
            to ensure mergability.
    """

    @property
    def primitive(self) -> str:  # noqa: D102  # No docstring needed.
        return "dynamic_slice"

    @override
    def __call__(
        self,
        builder: translator.JaxprTranslationBuilder,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        assert in_var_names[0]
        assert len(in_var_names) == len(util.get_jax_var_shape(eqn.invars[0])) + 1

        # This is the sizes of the slice window.
        window_sizes: Sequence[int] = eqn.params["slice_sizes"]

        # Maps the variable name, that stores the start index of the window in one
        #  dimensions to the access node, that holds the value. The variable name
        #  is also used as dynamic range offset.
        #  Only present if the index is not a literal.
        in_access: dict[str, dace.nodes.AccessNode] = {}

        # Name of the variable from where we get the start index of the window
        #  or the value itself, if it is a literal; in the order of the dimension.
        #  If the value is `None` then the literal was not yet processed.
        window_start_indices: list[str | None] = list(in_var_names[1:])

        # We will always adapt the start indexes and not check if it is needed.
        for dim, (window_start_index, dim_size, window_size) in enumerate(
            zip(window_start_indices, util.get_jax_var_shape(eqn.invars[0]), window_sizes)
        ):
            if window_start_index is None:
                # Jax does not adjust the literals on its own
                raw_window_start = int(util.get_jax_literal_value(eqn.invars[dim + 1]))  # type: ignore[arg-type]  # type confusion
                adjusted_window_start = min(dim_size, raw_window_start + window_size) - window_size
                window_start_indices[dim] = str(adjusted_window_start)
                continue

            # We do not use a symbol for the start of the window but a Tasklet, as
            #  a symbol would need an interstage edge, which is an optimization barrier.
            tasklet = dace.nodes.Tasklet(
                label=f"adjustment_of_slice_start_{window_start_index}_for_{out_var_names[0]}",
                inputs={"unadjusted_start_idx": None},
                outputs={"adjusted_start_idx": None},
                code=f"adjusted_start_idx = min(unadjusted_start_idx + {window_size}, {dim_size}) - {window_size}",
            )
            new_start_idx_var_name = builder.add_array(
                eqn.invars[dim + 1], name_prefix="__jace_adapted_start_idx_"
            )
            new_start_idx_acc = eqn_state.add_access(new_start_idx_var_name)

            eqn_state.add_edge(
                eqn_state.add_read(window_start_index),
                None,
                tasklet,
                "unadjusted_start_idx",
                dace.Memlet.simple(window_start_index, "0"),
            )
            eqn_state.add_edge(
                tasklet,
                "adjusted_start_idx",
                new_start_idx_acc,
                None,
                dace.Memlet.simple(new_start_idx_var_name, "0"),
            )
            # Update the name of the start index, and store the access
            #  node for later use.
            window_start_indices[dim] = new_start_idx_var_name
            in_access[new_start_idx_var_name] = new_start_idx_acc

        tskl_ranges: list[tuple[str, str]] = [
            (f"__i{dim}", f"0:{N}") for dim, N in enumerate(util.get_jax_var_shape(eqn.outvars[0]))
        ]

        memlet_accesses: list[str] = []

        for (it_var, _), offset_symbol_name in zip(tskl_ranges, window_start_indices):
            assert offset_symbol_name is not None
            memlet_accesses.append(f"{it_var} + {offset_symbol_name}")

        tskl_input = dace.Memlet.simple(in_var_names[0], ", ".join(memlet_accesses))
        tskl_output = dace.Memlet.simple(
            out_var_names[0], ", ".join(name for name, _ in tskl_ranges)
        )
        _, map_entry, _ = eqn_state.add_mapped_tasklet(
            name=f"{self.primitive}_{out_var_names[0]}",
            map_ranges=tskl_ranges,
            inputs={"__in": tskl_input},
            code="__out = __in",
            outputs={"__out": tskl_output},
            external_edges=True,
        )

        # Creating the inputs for the dynamic map ranges. We have to use the same
        #  access nodes as above, to ensure a single order of computation.
        for window_start_index_name, windows_start_access_node in in_access.items():
            eqn_state.add_edge(
                windows_start_access_node,
                None,
                map_entry,
                window_start_index_name,
                dace.Memlet.simple(window_start_index_name, "0"),
            )
            map_entry.add_in_connector(window_start_index_name)


translator.register_primitive_translator(SlicingTranslator())
translator.register_primitive_translator(DynamicSlicingTranslator())
