# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains the translation context for the `JaxprTranslationDriver`."""

from __future__ import annotations

from collections.abc import MutableMapping

import dace
from jax import core as jax_core

from jace import translator, util


class _TranslationContext:
    """Represents the context of a `JaxprTranslationDriver`.

    Essentially it contains the following variables:
    - `sdfg`:
        The SDFG object that is under construction.
    - `start_state`:
        The first state in the SDFG state machine.
    - `terminal_state`:
        The current terminal state of the SDFG state machine.
    - `jax_name_map`:
        A `dict` that maps every Jax variable to its corresponding SDFG variable _name_.
    - `inp_names`:
        A `list` of the SDFG variable names that are used for input.
        Their order is the same as in `Jaxpr.invars`.
        Filled at the very beginning.
    - `out_names`:
        A `list` of the SDFG variables names that are used for output,
        Their order is the same as in `Jaxpr.outvars`.
        Only filled at the very end.
    - `rev_idx`:
        The revision index (used to generate unique names in the translation.

    Notes:
        It might be that a name appears in both the `inp_names` and `out_names` list.
            This happens if the corresponding variable is used as both input and output.
            In Jax this is called argument donation.
        This class is similar to but different to `TranslatedJaxprSDFG`.
            This class is used to represent the dynamic state of the translation object,
            `TranslatedJaxprSDFG` is used to result the end.
    """

    __slots__ = (
        "sdfg",
        "start_state",
        "terminal_state",
        "jax_name_map",
        "inp_names",
        "out_names",
        "rev_idx",
    )

    def __init__(
        self,
        rev_idx: int,
        name: str | None = None,
    ) -> None:
        """Initializes the context.

        Args:
            rev_idx:    The revision index of the context.
            name:       Name of the SDFG object.
        """
        if isinstance(name, str) and not util.VALID_SDFG_OBJ_NAME.fullmatch(name):
            raise ValueError(f"'{name}' is not a valid SDFG name.")

        self.sdfg: dace.SDFG = dace.SDFG(name=(name or f"unnamed_SDFG_{id(self)}"))
        self.start_state: dace.SDFGState = self.sdfg.add_state(
            label="initial_state", is_start_block=True
        )
        self.terminal_state: dace.SDFGState = self.start_state
        self.jax_name_map: MutableMapping[jax_core.Var | util.JaCeVar, str] = {}
        self.inp_names: tuple[str, ...] = ()
        self.out_names: tuple[str, ...] = ()
        self.rev_idx: int = rev_idx

    def to_translated_jaxpr_sdfg(self) -> translator.TranslatedJaxprSDFG:
        """Transforms `self` into a `TranslatedJaxprSDFG`."""
        return translator.TranslatedJaxprSDFG(
            sdfg=self.sdfg,
            start_state=self.start_state,
            terminal_state=self.terminal_state,
            jax_name_map=self.jax_name_map,
            inp_names=self.inp_names,
            out_names=self.out_names,
        )
