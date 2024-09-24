# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translators for slicing operations."""

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

    The `slice` primitive represents the static case of slicing, i.e. a fixed
    window starting from a fixed starting point.
    The slicing is implemented by performing a partial copy.

    Note:
        Slices are essentially optimization barriers as they can not be fused
        with Maps _before_ them.
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
        strides: Sequence[int] = (
            ((1,) * len(tskl_ranges)) if eqn.params["strides"] is None else eqn.params["strides"]
        )
        start_indices: Sequence[int] = eqn.params["start_indices"]  # Fist index to slice
        return {
            "__in0": dace.Memlet.simple(
                in_var_names[0],
                ", ".join(
                    f"{start_index} + ({it_idx} * {stride})"
                    for (it_idx, _), start_index, stride in zip(tskl_ranges, start_indices, strides)
                ),
            )
        }


@translator.register_primitive_translator()
@translator.make_primitive_translator("dynamic_slice")
def dynamic_slicing_translator(
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `dynamic_slice` primitive.

    Dynamic slicing (see: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_slice.html)
    performs a slicing of a _fixed_ window, but the start of the window is defined
    through some input variables. Furthermore, if the window would overrun the
    start indexes are adjusted.

    Todo:
        - Prevent that the modified start indexes are promoted to symbols,
            to ensure mergability.
    """
    assert in_var_names[0]
    assert len(in_var_names) == len(util.get_jax_var_shape(eqn.invars[0])) + 1

    window_sizes: Sequence[int] = eqn.params["slice_sizes"]

    # Maps the variable name, that stores the _adjusted_ start index of the window
    #  of a dimension to the access node that holds the value. Needed to ensure the
    #  correct order of computation.
    in_access: dict[str, dace.nodes.AccessNode] = {}

    # Name of the variables (DaCe arrays) from where we get the start index of the
    #  window or the value itself if it is a literal (`None` means not yet processed).
    # The first input argument is always the array we slice from.
    window_start_indices: list[str | None] = list(in_var_names[1:])

    for dim, (window_start_index, dim_size, window_size) in enumerate(
        zip(window_start_indices, util.get_jax_var_shape(eqn.invars[0]), window_sizes)
    ):
        if window_start_index is None:
            # The start is a literal value.
            #  Jax does not adjust the literals on its own so we have to do it.
            raw_window_start = int(util.get_jax_literal_value(eqn.invars[dim + 1]))  # type: ignore[arg-type]  # type confusion
            adjusted_window_start = min(dim_size, raw_window_start + window_size) - window_size
            window_start_indices[dim] = str(adjusted_window_start)

        else:
            tasklet = dace.nodes.Tasklet(
                label=f"adjustment_of_slice_start_{window_start_index}_for_{out_var_names[0]}",
                inputs={"unadjusted_start_idx": None},
                outputs={"adjusted_start_idx": None},
                code=f"adjusted_start_idx = min(unadjusted_start_idx + {window_size}, {dim_size}) - {window_size}",
            )
            # Name of the variable holding the (adjusted) start of the window.
            #  It is important that this name is also used for the dynamic map range
            #  symbols created below. This prevents some errors if DaCe promotes them
            #  to symbols and does not handle the DMR correctly.
            #  (see https://github.com/spcl/dace/issues/1665)
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
            window_start_indices[dim] = new_start_idx_var_name
            in_access[new_start_idx_var_name] = new_start_idx_acc

    tskl_ranges: list[tuple[str, str]] = [
        (f"__i{dim}", f"0:{N}") for dim, N in enumerate(util.get_jax_var_shape(eqn.outvars[0]))
    ]
    tskl_input = dace.Memlet.simple(
        in_var_names[0],
        ", ".join(
            f"{it_var} + {offset_symbol_name}"
            for (it_var, _), offset_symbol_name in zip(tskl_ranges, window_start_indices)
        ),
    )
    tskl_output = dace.Memlet.simple(out_var_names[0], ", ".join(name for name, _ in tskl_ranges))
    _, map_entry, _ = eqn_state.add_mapped_tasklet(
        name=f"dynamic_slice_{out_var_names[0]}",
        map_ranges=tskl_ranges,
        inputs={"__in": tskl_input},
        code="__out = __in",
        outputs={"__out": tskl_output},
        external_edges=True,
    )

    # Create the dynamic ranges, i.e. read the start indexes for the window
    #  from variable and create symbols out of it, without an interstate edge.
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
