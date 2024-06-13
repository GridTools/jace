# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the translator related to data movement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from jace import translator
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class CopyTranslator(mapped_base.MappedOperationTranslatorBase):
    """
    Implements the `copy` primitive.

    Copy operations are implemented as a map to ensure that they can be fused
    with other maps
    .
    """

    def __init__(self) -> None:
        super().__init__(primitive_name="copy")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        return "__out = __in0"


class DevicePutTranslator(mapped_base.MappedOperationTranslatorBase):
    """
    Implements the `device_put` primitive.

    In Jax this primitive is used to copy data between the host and the device.
    Because of the way how JaCe and the optimization pipeline works, either
    everything is on the host or the device.

    Todo:
        Think about how to implement this correctly.
    """

    def __init__(self) -> None:
        super().__init__(primitive_name="device_put")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        if not (eqn.params["device"] is None and eqn.params["src"] is None):
            raise NotImplementedError(
                f"Can only copy on the host, but not from {eqn.params['src']} to {eqn.params['device']}."
            )
        return "__out = __in0"


_ = translator.register_primitive_translator(CopyTranslator())
_ = translator.register_primitive_translator(DevicePutTranslator())
