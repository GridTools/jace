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
        assert len(in_var_names) == len(util.get_jax_var_shape(eqn.invars[0])) + 1

        # This is the sizes of the slice window.
        window_sizes: Sequence[int] = eqn.params["slice_sizes"]

        # The first input to the primitive is the array we slice from, the others are the start
        #  indices of the slice window, each is a scalar, maybe literals, we might adapt them later.
        in_var_name: str = in_var_names[0]
        start_indices: list[str | None] = list(in_var_names[1:])

        # For storing the adapted start index, we have to create access nodes, to store them.
        #  However, to ensure a total order of execution, once we added them as dynamic map ranges
        #  to the map, we must use the same access nodes.
        in_access: dict[str, dace.nodes.AccessNode] = {}

        # Jax will adjust the start indexes if the window will overrun.
        #  The adjustment is based on the formula $min(s + w, N) - w$, where $s$ is the start
        #  index, $w$ the window size and $N$ the length in a particular dimension.
        #  To do it we will use Tasklets, because otherwise we can not merge the state.
        # TODO(phimuell): Make the Tasklet mapped, that they can be merged.
        for dim, (start_index, dim_size, wsize) in enumerate(
            zip(start_indices, util.get_jax_var_shape(eqn.invars[0]), window_sizes)
        ):
            if start_index is None:
                continue

            tasklet = dace.nodes.Tasklet(
                label=f"adjustment_of_slice_start_{start_index}_for_{out_var_names[0]}",
                inputs={"unadjusted_start_idx": None},
                outputs={"adjusted_start_idx": None},
                code=f"adjusted_start_idx = min(unadjusted_start_idx + {wsize}, {dim_size}) - {wsize}",
            )

            # Intermediate value for the adjusted start index.
            new_start_idx_var_name = driver.add_array(
                eqn.invars[dim + 1],
                name_prefix=f"__jace_adapted_start_idx_{start_index}",
            )
            new_start_idx_acc = eqn_state.add_access(new_start_idx_var_name)

            # Create the connections to and from the Tasklet.
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

            # Now store the result
            start_indices[dim] = new_start_idx_var_name
            in_access[new_start_idx_var_name] = new_start_idx_acc

        tskl_ranges: list[tuple[str, str]] = [
            (f"__i{dim}", f"0:{N}") for dim, N in enumerate(util.get_jax_var_shape(eqn.outvars[0]))
        ]

        # We use dynamic map ranges, thus the map entry has input connectors, that does not start
        #  with `IN_*`, instead the connector name defines a symbol within the map scope. This
        #  `dict` maps the symbol name to the name of the input variable, that defines the symbol.
        #  If the input is a literal, than it has no correspondence and the constant is substituted.
        dynamic_map_ranges: dict[str, str] = {}
        memlet_accesses: list[str] = []

        for i, ((it_var, _), start_index) in enumerate(zip(tskl_ranges, start_indices)):
            if start_index is None:
                offset = str(util.get_jax_literal_value(eqn.invars[i + 1]))
            else:
                # Because of [issue 1579](https://github.com/spcl/dace/issues/1579) we have to use
                #  the same name as the data container for the symbol and can not mangle it.
                # TODO(phimuell): Activate mangling when the issue is resolved.
                # offset = f"__jace_dynamic_map_range_{out_var_names[0]}_{start_index}" # noqa: ERA001
                offset = start_index
                dynamic_map_ranges[offset] = start_index
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
