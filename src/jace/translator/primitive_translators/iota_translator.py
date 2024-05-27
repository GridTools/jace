# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This implements the `iota` primitive."""

from __future__ import annotations

from collections.abc import Sequence

import dace
from jax import core as jax_core
from typing_extensions import override

from jace import translator
from jace.translator import mapped_operation_base_translator as mapped_base


class IotaTranslator(mapped_base.MappedOperationTranslatorBase):
    """This handles the `iota` primitives.

    Essentially a very general `jnp.arange()` function.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(primitive_name="iota")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        return f"__out = {tskl_ranges[eqn.params['dimension']][0]}"

    @override
    def make_input_memlets(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> dict[str, dace.Memlet]:
        return {}


translator.register_primitive_translator(IotaTranslator())
