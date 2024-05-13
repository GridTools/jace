# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass

import dace
from jax import core as jax_core

from jace import util


@dataclass(slots=True)
class TranslatedJaxprSDFG:
    """Encapsulates the result of a translation run of the `JaxprTranslationDriver` object.

    This class is also used to represent the internal state of the `JaxprTranslationDriver` during the translation.
    For that reason the object defines some fields that only have a meaning during the actually translation.

    The fields used to store the result are:
    - `sdfg` the SDFG object that was created.
    - `jax_name_map` a `dict` that maps every Jax variable to its corresponding SDFG variable _name_.
    - `start_state` the first state in the SDFG state machine.
    - `terminal_state` the last state in the state machine.
    - `inp_names` a `list` of the SDFG variables that are used as input, in the same order as `Jaxpr.invars`.
    - `out_names` a `list` of the SDFG variables that are used as output, in the same order as `Jaxpr.outvars`.

    Please consider the following important points:
    - The SDFG is in canonical form, which means that it is not directly usable, see `JaxprTranslationDriver` for more.
    - It might be that a name appears in both the `inp_names` and `out_names` list.
        This happens if the corresponding variable is used as both input and output.
        In Jax this is called argument donation.

    During the translation the following members are also allocated:
    - `rev_idx` the revision index, used for name mangling.

    While they remain allocated, accessing them is considered an error.
    """

    sdfg: dace.SDFG
    jax_name_map: MutableMapping[jax_core.Var | util.JaCeVar, str]
    start_state: dace.SDFGState
    terminal_state: dace.SDFGState
    inp_names: tuple[str, ...]
    out_names: tuple[str, ...]
    rev_idx: int

    def __init__(
        self,
        rev_idx: int,
        name: str | None = None,
    ) -> None:
        """Initializes the context.

        The function allocates the SDFG and initializes the members properly.

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

    def validate(self) -> bool:
        """Validate the underlying SDFG."""

        # To prevent the 'non initialized' data warnings we have to temporary
        #  promote input and output arguments to globals
        org_trans_state: dict[str, bool] = {}
        if not self.inp_names:
            raise dace.sdfg.InvalidSDFGError(
                "There are no input arguments.",
                self.sdfg,
                self.sdfg.node_id(self.start_state),
            )
        if not self.out_names:
            raise dace.sdfg.InvalidSDFGError(
                "There are no output arguments.",
                self.sdfg,
                self.sdfg.node_id(self.start_state),
            )
        for var in set(self.inp_names + self.out_names):  # set is needed for donated args.
            org_trans_state[var] = self.sdfg.arrays[var].transient
            self.sdfg.arrays[var].transient = False

        try:
            self.sdfg.validate()
        finally:
            for var, orgValue in org_trans_state.items():
                self.sdfg.arrays[var].transient = orgValue
        return True
