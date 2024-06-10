# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import dace
from typing_extensions import override

from jace import translator, util
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class SqueezeTranslator(mapped_base.MappedOperationTranslatorBase):
    """Implements the `squeeze` primitive.

    The primitives allows to remove a dimension of size one. Essentially
    equivalent to `np.squeeze` and the inverse to `np.expand_dims()`,
    which is handled by the `broadcast_in_dim` primitive.
    """

    def __init__(self) -> None:
        super().__init__(primitive_name="squeeze")

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
        dims_to_delete: Sequence[str] = eqn.params["dimensions"]
        in_rank: int = len(util.get_jax_var_shape(eqn.invars[0]))
        cnt = itertools.count(0)
        return {
            "__in0": dace.Memlet.simple(
                in_var_names[0],
                ", ".join(
                    "0" if dim in dims_to_delete else tskl_ranges[next(cnt)][0]
                    for dim in range(in_rank)
                ),
            )
        }


translator.register_primitive_translator(SqueezeTranslator())
