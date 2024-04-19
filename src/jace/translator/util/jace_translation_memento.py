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


@dataclass(init=True, repr=True, eq=False, frozen=True, kw_only=True, slots=True)
class JaCeTranslationMemento:
    """Encapsulates the result of a translation run of the 'JaxprTranslationDriver' object.

    It defines the following members:
    - 'sdfg' the SDFG object that was created.
    - 'start_state' the first state in the SDFG state machine.
    - 'terminal_state' the last state in the state machine.
    - 'jax_name_map' a 'dict' that maps every Jax name to its corresponding SDFG variable name.
    - 'inp_names' a 'list' of the SDFG variables that are used as input, in the same order as 'Jaxpr.invars'.
    - 'out_names' a 'list' of the SDFG variables that are used as output, in the same order as 'Jaxpr.outvars'.
    """

    sdfg: dace.SDFG
    start_state: dace.SDFGState
    terminal_state: dace.SDFGState
    jax_name_map: Mapping[str, str]
    inp_names: Sequence[str]
    out_names: Sequence[str]

    def validate(self) -> bool:
        """Validate the underlying SDFG."""

        # To prevent the 'non initialized' data warnings we have to temporary promote the input arguments as global.
        org_trans_state: dict[str, bool] = {}
        for var in self.inp_names:
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

    def __hash__(self) -> int:
        """Computes the hash of the underlying SDFG object."""
        return hash(self.sdfg)

    def __eq__(self, other: Any) -> bool:
        """Compares the underlying SDFG object with 'rhs'."""
        if isinstance(other, JaCeTranslationMemento):
            return bool(self.sdfg == other.sdfg)
        elif hasattr(other, "__sdfg__"):
            other = other.__sdfg__()
        elif isinstance(other, dace.SDFG):
            pass
        else:
            return NotImplemented
        #

        x: bool = self.sdfg.__eq__(other)
        return x
