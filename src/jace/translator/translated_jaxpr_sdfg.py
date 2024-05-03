# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import dace
from jax import core as jax_core

from jace import util


@dataclass(init=True, repr=True, eq=False, frozen=False, kw_only=True, slots=True)
class TranslatedJaxprSDFG:
    """Encapsulates the result of a translation run of the `JaxprTranslationDriver` object.

    It defines the following members:
    - `sdfg` the SDFG object that was created.
    - `jax_name_map` a `dict` that maps every Jax variable to its corresponding SDFG variable _name_.
    - `start_state` the first state in the SDFG state machine.
    - `terminal_state` the last state in the state machine.
    - `inp_names` a `list` of the SDFG variables that are used as input, in the same order as `Jaxpr.invars`.
    - `out_names` a `list` of the SDFG variables that are used as output, in the same order as `Jaxpr.outvars`.

    The SDFG is in a so called canonical form, that is not directly usable, see `JaxprTranslationDriver` for more.

    It might be that a name appears in both the `inp_names` and `out_names` list.
    This happens if the corresponding variable is used as both input and output.
    In Jax this is called argument donation.
    """

    sdfg: dace.SDFG
    jax_name_map: Mapping[jax_core.Var | util.JaCeVar, str]
    start_state: dace.SDFGState | None = None
    terminal_state: dace.SDFGState | None = None
    inp_names: Sequence[str] | None = None
    out_names: Sequence[str] | None = None

    def validate(self) -> bool:
        """Validate the underlying SDFG."""

        # To prevent the 'non initialized' data warnings we have to temporary
        #  promote input and output arguments to globals
        promote_to_glob: set[str] = set()
        org_trans_state: dict[str, bool] = {}
        if self.inp_names:
            promote_to_glob.update(self.inp_names)
        if self.out_names:
            promote_to_glob.update(self.out_names)
        for var in promote_to_glob:
            org_trans_state[var] = self.sdfg.arrays[var].transient
            self.sdfg.arrays[var].transient = False

        try:
            self.sdfg.validate()
        finally:
            for var, orgValue in org_trans_state.items():
                self.sdfg.arrays[var].transient = orgValue
        return True

    def __getitem__(self, idx: str) -> Any:
        """Allows member access using brackets."""
        if not isinstance(idx, str):
            raise TypeError(f"Expected 'idx' as 'str' but got '{type(str)}'")
        if not hasattr(self, idx):
            raise KeyError(f"The key '{idx}' is not known.")
        return getattr(self, idx)
