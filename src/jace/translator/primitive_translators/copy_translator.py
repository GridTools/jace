# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translators related to data movement operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace

from jace import translator


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


@translator.register_primitive_translator()
@translator.make_primitive_translator("copy")
def copy_translator(
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,  # noqa: ARG001  # Required by the interface.
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `copy` primitive.

    Todo:
        Investigate if operation should expand to a map.
    """
    eqn_state.add_nedge(
        eqn_state.add_read(in_var_names[0]),
        eqn_state.add_write(out_var_names[0]),
        dace.Memlet.from_array(
            in_var_names[0],
            builder.arrays[in_var_names[0]],  # type: ignore[index]  # Guaranteed to be a string
        ),
    )


@translator.register_primitive_translator()
@translator.make_primitive_translator("device_put")
def device_put_translator(
    builder: translator.JaxprTranslationBuilder,
    in_var_names: Sequence[str | None],
    out_var_names: Sequence[str],
    eqn: jax_core.JaxprEqn,
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `device_put` primitive.

    In JAX this primitive is used to copy data between the host and the device,
    in DaCe Memlets can do this. However, because of the way JaCe operates, at
    least in the beginning a computation is either fully on the host or on the
    device this copy will essentially perform a copying.
    """
    if not (eqn.params["device"] is None and eqn.params["src"] is None):
        raise NotImplementedError(
            f"Can only copy on the host, but not from {eqn.params['src']} to {eqn.params['device']}."
        )
    copy_translator(
        builder=builder,
        in_var_names=in_var_names,
        out_var_names=out_var_names,
        eqn=eqn,
        eqn_state=eqn_state,
    )
