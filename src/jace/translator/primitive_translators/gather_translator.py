# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the translator for the `gather` primitive."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace
from jax import lax as jax_lax
from typing_extensions import override

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class GatherTranslator(translator.PrimitiveTranslator):
    """
    Garther Translator.

    The gather operation extracts patches of a certain size, known as `slice_size`,
    from an array, called source or input array. Where these patches starts is
    given by another array, the index array.

    See Also:
        https://www.tensorflow.org/xla/operation_semantics#gather
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.gather.html
    """

    @property
    def primitive(self) -> str:  # noqa: D102  # No docstring needed.
        return "gather"

    @override
    def __call__(  # noqa: PLR0914, PLR0915  # Just ported from the prototype, cleanup postponed.
        self,
        builder: translator.JaxprTranslationBuilder,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        """
        Performs the gather operation.

        Args:
            builder: The builder object that is active.
            in_var_names: The names of the input variables, the first array is
                assumed as source array and the second is the index array.
            out_var_names: The names of the output variables.
            eqn: The equation to translate.
            eqn_state: The state in which we put the extraction.
        """
        assert len(eqn.invars) == 2  # noqa: PLR2004  # XLA supports more inputs.

        out_name = out_var_names[0]
        out_shape = util.get_jax_var_shape(eqn.outvars[0])

        src_name = in_var_names[0]
        src_shape = util.get_jax_var_shape(eqn.invars[0])

        idx_name = in_var_names[1]
        idx_shape = util.get_jax_var_shape(eqn.invars[1])

        dimension_numbers = eqn.params["dimension_numbers"]
        offset_dims: Sequence[int] = dimension_numbers.offset_dims
        collapsed_slice_dims: Sequence[int] = dimension_numbers.collapsed_slice_dims
        start_index_map: Sequence[int] = dimension_numbers.start_index_map
        slice_sizes: Sequence[int] = eqn.params["slice_sizes"]
        mode: jax_lax.GatherScatterMode = eqn.params["mode"]
        assert len(start_index_map) == idx_shape[-1]

        if mode != jax_lax.GatherScatterMode.PROMISE_IN_BOUNDS:
            raise NotImplementedError(f"The mode {mode} is not implemented.")

        # Over these dimensions the copy of the patches goes.
        batch_dims = tuple(d for d in range(len(out_shape)) if d not in offset_dims)

        # Every batch dimension is associated with one dimension of of the index
        #  array, but there is always one dimension more in the index array. This
        #  dimension contains the start indexes of the slice, if there is only
        #  one index that should be loaded is not strictly necessary, but Jax
        #  (currently adds) it implicitly, probably to make life easier.
        if (len(batch_dims) + 1) != len(idx_shape):
            raise ValueError(
                f"Expected that the index array has {len(batch_dims) + 1} dimensions, but it had {len(idx_shape)}."
            )

        # These are the dimensions (of the input) for which a map index is created.
        #  Note that we exclude collapsed dimensions here.
        src_dim_with_map_idx = tuple(
            dim for dim in range(len(slice_sizes)) if dim not in collapsed_slice_dims
        )
        assert len(src_dim_with_map_idx) == len(offset_dims)

        # The final map is the composition of two loops. The first map iterates over
        #  the index array, except the last dimension, and is used to "copy" the
        #  different patches from the source to the output array. These map parameters
        #  follow the pattern `__i{out_name}_gather{bd}`, where `bd` is a batch
        #  dimension. These variables are used to access the index array.
        #  The second loop performs the actual copy of the slices. For these
        #  the variables `__i{i}` is used were, these are known as offset
        #  dimensions.
        #  What is a bit difficult, that the actual access/dereferencing of the source
        #  array is done within the tasklet.

        # Access pattern of the source array _within_ the tasklet.
        src_access_pattern: list[str] = []

        # These are the map ranges for the coying of the slicing.
        slice_map_ranges: list[tuple[str, str]] = []

        # Compute the access pattern within the tasklet.
        #  As a side effect we also compute the map ranges, but only for the slices.
        for dim, slice_size in enumerate(slice_sizes):
            # Order is important!
            if dim not in start_index_map:
                # This dimension is fully copied
                slice_map_ranges.append((f"__i{dim}", f"0:{slice_size}"))
                src_access_pattern.append(slice_map_ranges[-1][0])
                assert dim in src_dim_with_map_idx
                assert slice_size == src_shape[dim]

            elif dim in collapsed_slice_dims:
                # This dimension is only partially copied, however, since the
                #  dimension is collapsed, only a single element is copied that
                #  comes from the index array.
                src_access_pattern.append(f"__gather_{dim}")

            else:
                # This dimension is partially copied, but is _not colapsed_, we need
                #  a map index to copy the range. However, there is also an offset
                #  that is involved from copying.
                slice_map_ranges.append((f"__i{dim}", f"0:{slice_size}"))
                src_access_pattern.append(f"__gather_{dim} + {slice_map_ranges[-1][0]}")
                assert dim in src_dim_with_map_idx
                assert slice_size <= src_shape[dim]

        # These are the map variable that go over the index array.
        patch_loop_vars = tuple(f"__i{out_name}_gather{bd}" for bd in batch_dims)
        patch_map_ranges = [
            (map_param, f"0:{patch_loop_bound}")
            for map_param, patch_loop_bound in zip(patch_loop_vars, idx_shape[:-1])
        ]

        # Creating the input memlet that allows us to access the source array from
        #  inside the tasklet and make it accessible through the name `__arr`. At
        #  this point it is not possible to tell where we access, because we are
        #  missing a index variables, they will only be accessible inside the
        #  tasklet (see below), however, we know that we will access only one
        #  element from the array.
        tasklet_inputs: dict[str, dace.Memlet] = {
            "__arr": dace.Memlet.simple(
                data=src_name,
                subset_str=", ".join(f"0:{size}" for size in src_shape),
                num_accesses=1,
            ),
        }

        # Now we are creating the memlets that access the index array.
        for i, dim in enumerate(start_index_map):
            tasklet_inputs[f"__gather_{dim}"] = dace.Memlet.simple(
                data=idx_name, subset_str=(", ".join(patch_loop_vars) + f", {i}")
            )

        # The tasklet code.
        tasklet_code = "__out = __arr[" + ", ".join(src_access_pattern) + "]"

        # Now we generate the output memlet.
        outpt_access_pattern: list[str] = []
        dim_counter = 0
        for dim in range(len(out_shape)):
            if dim in batch_dims:
                # This is a batch dimension, thus a loop variable is used for it.
                patch_loop_var = patch_loop_vars[batch_dims.index(dim)]
                outpt_access_pattern.append(str(patch_loop_var))

            else:
                # This is a dimension for copying the slices.
                assert dim_counter <= len(src_dim_with_map_idx)
                associated_map_idx = src_dim_with_map_idx[dim_counter]
                dim_counter += 1
                outpt_access_pattern.append(f"__i{associated_map_idx}")
        assert dim_counter == len(src_dim_with_map_idx)

        tasklet_outputs: dict[str, dace.Memlet] = {
            "__out": dace.Memlet.simple(data=out_name, subset_str=", ".join(outpt_access_pattern))
        }
        assert len(patch_map_ranges) + len(slice_map_ranges) == len(out_shape)

        eqn_state.add_mapped_tasklet(
            name=f"_gather_map_{out_name}",
            map_ranges=patch_map_ranges + slice_map_ranges,
            inputs=tasklet_inputs,
            code=tasklet_code,
            outputs=tasklet_outputs,
            external_edges=True,
        )


_ = translator.register_primitive_translator(GatherTranslator())
