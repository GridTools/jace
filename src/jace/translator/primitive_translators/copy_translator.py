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
    eqn: jax_core.JaxprEqn,  # noqa: ARG001 [unused-function-argument]  # Required by the interface.
    eqn_state: dace.SDFGState,
) -> None:
    """
    Implements the `copy` primitive.

    The copy is implemented by creating a memlet between the source and destination.

    Args:
        builder: The builder object of the translation.
        in_var_names: The SDFG variable that acts as source.
        out_var_names: The SDFG variable that acts as destination of the copy.
        eqn: The equation that should be translated; unused.
        eqn_state: State into which the nested SDFG should be constructed.

    Todo:
        Investigate if operation should expand to a map.
    """
    assert in_var_names[0] is not None
    eqn_state.add_nedge(
        eqn_state.add_read(in_var_names[0]),
        eqn_state.add_write(out_var_names[0]),
        dace.Memlet.from_array(
            in_var_names[0],
            builder.arrays[in_var_names[0]],
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
    in DaCe only memlets can do this. However, because of the way JaCe (currently)
    operates (a computation is either fully on the host or on GPU), the `device_put`
    primitive essentially decays to a copy.

    Args:
        builder: The builder object of the translation.
        in_var_names: The SDFG variable that acts as source.
        out_var_names: The SDFG variable that acts as destination of the copy.
        eqn: The equation that should be translated.
        eqn_state: State into which the nested SDFG should be constructed.
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
