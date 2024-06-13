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
    from a fixed initial position. The `dynamic_slice` operation supports a
    variable starting point.
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

        # The first input to the primitive is the array we slice from, the others are
        #  the start indices of the slice window, each is a scalar, maybe literals.
        in_var_name: str = in_var_names[0]
        start_indices: list[str | None] = list(in_var_names[1:])

        # Access nodes for the modified start indexes.
        in_access: dict[str, dace.nodes.AccessNode] = {}

        # We will always adapt the start indexes and not check if it is needed.
        for dim, (start_index, dim_size, wsize) in enumerate(
            zip(start_indices, util.get_jax_var_shape(eqn.invars[0]), window_sizes)
        ):
            if start_index is None:
                continue

            # We use a Tasklet to perform the adjustment not a symbol, because this
            # would need an interstage edge serving as kind of an optimization barrier.
            tasklet = dace.nodes.Tasklet(
                label=f"adjustment_of_slice_start_{start_index}_for_{out_var_names[0]}",
                inputs={"unadjusted_start_idx": None},
                outputs={"adjusted_start_idx": None},
                code=f"adjusted_start_idx = min(unadjusted_start_idx + {wsize}, {dim_size}) - {wsize}",
            )

            new_start_idx_var_name = builder.add_array(
                eqn.invars[dim + 1], name_prefix="__jace_adapted_start_idx_"
            )
            new_start_idx_acc = eqn_state.add_access(new_start_idx_var_name)

            eqn_state.add_edge(
                eqn_state.add_read(start_index),
                None,
                tasklet,
                "unadjusted_start_idx",
                dace.Memlet.simple(start_index, "0"),
            )
            eqn_state.add_edge(
                tasklet,
                "adjusted_start_idx",
                new_start_idx_acc,
                None,
                dace.Memlet.simple(new_start_idx_var_name, "0"),
            )
            # Update the name of the start index
            start_indices[dim] = new_start_idx_var_name
            in_access[new_start_idx_var_name] = new_start_idx_acc

        tskl_ranges: list[tuple[str, str]] = [
            (f"__i{dim}", f"0:{N}") for dim, N in enumerate(util.get_jax_var_shape(eqn.outvars[0]))
        ]

        # For copying the data, we use dynamic map ranges, which is basically an input
        #  connector on the map entry whose name is not `IN_*`, this name can then be
        #  used as a symbol inside the map scope; this symbol is then used as offset.
        dynamic_map_ranges: dict[str, str] = {}
        memlet_accesses: list[str] = []

        for i, ((it_var, _), start_index) in enumerate(zip(tskl_ranges, start_indices), 1):
            if start_index is None:
                offset = str(util.get_jax_literal_value(eqn.invars[i]))
            else:
                # Because of [issue 1579](https://github.com/spcl/dace/issues/1579) we
                #  have to use the same name as the data container for the symbol and
                #  can not mangle it.
                # TODO(phimuell): Activate mangling when the issue is resolved.
                offset = start_index
                dynamic_map_ranges[offset] = start_index
            memlet_accesses.append(f"{it_var} + {offset}")

        tskl_input = dace.Memlet.simple(in_var_name, ", ".join(memlet_accesses))
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
        for symb_name, start_index in dynamic_map_ranges.items():
            eqn_state.add_edge(
                in_access[start_index],
                None,
                map_entry,
                symb_name,
                dace.Memlet.simple(start_index, "0"),
            )
            map_entry.add_in_connector(symb_name)


translator.register_primitive_translator(SlicingTranslator())
translator.register_primitive_translator(DynamicSlicingTranslator())
