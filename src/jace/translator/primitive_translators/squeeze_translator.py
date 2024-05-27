# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from collections.abc import Sequence

import dace
from jax import core as jax_core
from typing_extensions import override

from jace import translator
from jace.translator import mapped_operation_base_translator as mapped_base


class SqueezeTranslator(mapped_base.MappedOperationTranslatorBase):
    """Allows to remove dimensions with size one.

    Essentially equivalent to `np.squeeze` and the inverse to `np.expand_dims()`.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(primitive_name="squeeze")

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
        to_rem: Sequence[str] = eqn.params["dimensions"]
        in_rank: int = len(eqn.invars[0].aval.shape)
        cnt = itertools.count(0)
        return {
            "__in0": dace.Memlet.simple(
                in_var_names[0],
                ", ".join(
                    "0" if dim in to_rem else tskl_ranges[next(cnt)][0] for dim in range(in_rank)
                ),
            )
        }


translator.register_primitive_translator(SqueezeTranslator())
