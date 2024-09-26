# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator for indexing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace
from jax import lax as jax_lax

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("gather")
def gather_translator(  # noqa: PLR0914 [too-many-locals]  # Can not reduce any further.
    builder: translator.JaxprTranslationBuilder,  # noqa: ARG001 [unused-function-argument]  # Required by the interface.
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `gather` primitive.

    These primitive is used to implement the `array.at[...].get()` access. In the
    end the primitive extracts patches/windows of a certain size, known as
    `slice_size`, from an array, which is called source or input array. The start
    points of these windows are given by another array, the so called index array.

    Args:
        builder: The builder object that is active.
        in_var_names: The names of the input variables, the first array is
            assumed as source array and the second is the index array.
        out_var_names: The names of the output variables.
        eqn: The equation to translate.
        eqn_state: The state in which we put the extraction.

    See Also:
        https://www.tensorflow.org/xla/operation_semantics#gather
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.gather.html
    """
    out_name = out_var_names[0]
    out_shape = util.get_jax_var_shape(eqn.outvars[0])
    src_name = in_var_names[0]
    src_shape = util.get_jax_var_shape(eqn.invars[0])
    idx_name = in_var_names[1]
    idx_shape = util.get_jax_var_shape(eqn.invars[1])
    dimension_numbers = eqn.params["dimension_numbers"]

    if eqn.params["mode"] != jax_lax.GatherScatterMode.PROMISE_IN_BOUNDS:
        raise NotImplementedError(f"The mode {eqn.params['mode']} is not implemented.")

    # This is the size of the slice window that is copied. Its length is the rank
    #  of the source array, dimensions that are excluded from copying are listed
    #  in `collapsed_slice_dims`.
    slice_sizes: Sequence[int] = eqn.params["slice_sizes"]
    collapsed_slice_dims: Sequence[int] = dimension_numbers.collapsed_slice_dims
    not_collapsed_slice_dims = tuple(
        dim for dim in range(len(slice_sizes)) if dim not in collapsed_slice_dims
    )
    assert len(slice_sizes) == len(src_shape)

    # The batch dimensions are used to iterate through the different slice windows
    #  (not inside them) thus they access the index array, with the exception of the
    #  last dimension, see below.
    # NOTE: In pure XLA this last dimension is in certain cases optional, however,
    #   JAX adds it and our implementation relies on it.
    batch_dims = tuple(d for d in range(len(out_shape)) if d not in dimension_numbers.offset_dims)
    if (len(batch_dims) + 1) != len(idx_shape):
        raise ValueError(
            f"Expected that the index array has {len(batch_dims) + 1} dimensions, but it had {len(idx_shape)}."
        )

    # The last dimension of the index array is special, as it contains the actual
    #  start point for the slice windows when the dimension is only partially copied.
    #  Thus the last dimension must be seen as a list of start indexes and the other
    #  dimensions are used to enumerate the slice windows. The `start_index_map`
    #  associates each position in the last dimension with the corresponding
    #  dimension of the source array.
    start_index_map: Sequence[int] = dimension_numbers.start_index_map
    assert len(start_index_map) == idx_shape[-1]

    # The iteration variable of the final map can be divided into two parts or
    #  categories. The first part iterates through all the slice windows that are
    #  given through the index array. If a dimension is not fully copied then the
    #  start index of the window is given through the elements of the last dimensions
    #  of the index array. Map variables that are used for this use the pattern
    #  `__i{out_name}_gather{bd}`. The second kind of variables are used to copy the
    #  content of the slice windows themselves, these map variables follow the
    #  pattern `__i{i}`.

    # Because the offsets of the slice window (which are given by the elements of
    #  the last dimension of the index array) are variables and not symbols, they
    #  can not be included in the memlets. Instead we generate a tasklet that
    #  performs an indirect access and get all elements of the last dimension of the
    #  index array (with the names `__gather_{dim}`), together with the full source
    #  array as input.

    # Access pattern of the source array _inside_ the tasklet.
    src_access_pattern: list[str] = []

    # The map variables and their ranges of the second part implicit loop; the one
    #  that copy the content inside the window.
    inside_window_map_ranges: list[tuple[str, str]] = []

    for dim, slice_size in enumerate(slice_sizes):
        # Order is important!
        if dim not in start_index_map:
            # This dimension is fully copied
            inside_window_map_ranges.append((f"__i{dim}", f"0:{slice_size}"))
            src_access_pattern.append(inside_window_map_ranges[-1][0])
            assert dim in not_collapsed_slice_dims

        elif dim in collapsed_slice_dims:
            # This dimension is only partially copied, but because it is collapsed,
            #  only a single element is copied. Thus the offset is only given by the
            #  what we read from the index array.
            src_access_pattern.append(f"__gather_{dim}")

        else:
            # This dimension is partially copied, but _not colapsed_. This the element
            #  that is read depends on the (static) offset of this window and the
            #  current position within the slicing window.
            inside_window_map_ranges.append((f"__i{dim}", f"0:{slice_size}"))
            src_access_pattern.append(f"__gather_{dim} + {inside_window_map_ranges[-1][0]}")
            assert dim in not_collapsed_slice_dims

    # These are the map variables that are associated to the first implicit loop (the
    #  iteration over the index array, excluding the last dimension).
    batch_map_ranges = [
        (f"__i{out_name}_gather{batch_dim}", f"0:{batch_loop_bound}")
        for batch_dim, batch_loop_bound in zip(batch_dims, idx_shape[:-1])
    ]
    assert len(batch_map_ranges) + len(inside_window_map_ranges) == len(out_shape)

    tasklet_inputs: dict[str, dace.Memlet] = {}

    # We need to pass the full array into the tasklet, however, we know that we
    #  will read only one element.
    tasklet_inputs["__arr"] = dace.Memlet.simple(
        data=src_name,
        subset_str=", ".join(f"0:{size}" for size in src_shape),
        num_accesses=1,
    )

    # The static offsets of the slice window, are given through the elements of the
    #  last dimensions of the index array.
    for i, dim in enumerate(start_index_map):
        tasklet_inputs[f"__gather_{dim}"] = dace.Memlet.simple(
            data=idx_name,
            subset_str=(
                ", ".join(batch_loop_var for batch_loop_var, _ in batch_map_ranges) + f", {i}"
            ),
        )

    # The output shape is given by the combination of the not collapsed slice sizes
    #  and the index array (without the last dimension) with some permutation.
    #  While the relative order of slice window does not change, `start_index_map`
    #  already applied a permutation, it might be interleaved with batch dimensions.
    output_memlet_pattern: list[str] = []
    dim_counter = 0
    for dim in range(len(out_shape)):
        if dim in batch_dims:
            batch_loop_var = batch_map_ranges[batch_dims.index(dim)][0]
            output_memlet_pattern.append(str(batch_loop_var))

        else:
            associated_map_idx = not_collapsed_slice_dims[dim_counter]
            dim_counter += 1
            output_memlet_pattern.append(f"__i{associated_map_idx}")
    assert dim_counter == len(not_collapsed_slice_dims)

    eqn_state.add_mapped_tasklet(
        name=f"_gather_map_{out_name}",
        map_ranges=batch_map_ranges + inside_window_map_ranges,
        inputs=tasklet_inputs,
        code="__out = __arr[" + ", ".join(src_access_pattern) + "]",
        outputs={
            "__out": dace.Memlet.simple(data=out_name, subset_str=", ".join(output_memlet_pattern))
        },
        external_edges=True,
    )
