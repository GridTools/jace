# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import dace
from typing_extensions import override

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class ReshapeTranslator(translator.PrimitiveTranslator):
    """Reshapes an array.

    Todo:
        - Handle `dimensions` parameter fully.
        - Find a way to make it as a Map.
    """

    @property
    def primitive(self) -> str:
        return "reshape"

    @override
    def __call__(
        self,
        builder: translator.JaxprTranslationBuilder,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        """Performs the reshaping.

        Currently a copy using a Memlet is performed.
        """
        if eqn.params["dimensions"] is not None:
            raise NotImplementedError("Currently 'dimensions' must be 'None'.")
        eqn_state.add_nedge(
            eqn_state.add_read(in_var_names[0]),
            eqn_state.add_write(out_var_names[0]),
            dace.Memlet(
                data=in_var_names[0],
                subset=", ".join(f"0:{size}" for size in util.get_jax_var_shape(eqn.invars[0])),
                other_subset=", ".join(f"0:{size}" for size in eqn.params["new_sizes"]),
            ),
        )


translator.register_primitive_translator(ReshapeTranslator())
