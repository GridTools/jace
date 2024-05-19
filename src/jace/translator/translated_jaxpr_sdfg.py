# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dace

from jace import util


class TranslatedJaxprSDFG:
    """Encapsulates the result of a translation run of the `JaxprTranslationDriver` object.

    The fields used to store the result are:
    - `sdfg` the SDFG object that was created.
    - `inp_names` a list of the SDFG variables that are used as input, in the same order as `Jaxpr.invars`.
    - `out_names` a list of the SDFG variables that are used as output, in the same order as `Jaxpr.outvars`.
    - `start_state` the first state in the SDFG state machine.
    - `terminal_state` the last state in the state machine.
    - `is_finalized` a bool that indicates if `self` represents a finalized or canonical SDFG, see below.

    Note, that it might happen that a name appears in both the `inp_names` and `out_names` lists.
    This happens if an argument is used both as input and output, and it is not an error.
    In Jax this is called argument donation.

    By default `self` encapsulates a canonical SDFG, see `JaxprTranslationDriver` for more information on this.
    However, if `is_finalized` is set, then `self` contains a finalized SDFG, i.e.
    - all input an output arrays are marked as global,
    - however, there are no `__return` arrays, i.e. all arguments are passed as arguments,
    - its `arg_names` are set with set `inp_names + out_names`, however,
        arguments that are input and outputs are only listed as inputs.

    Furthermore, only `sdfg`, `inp_names` and `out_names` are guaranteed to be allocated, all other fields might be `None`.
    """

    sdfg: dace.SDFG
    inp_names: tuple[str, ...]
    out_names: tuple[str, ...]
    is_finalized: bool
    start_state: dace.SDFGState | None
    terminal_state: dace.SDFGState | None

    def __init__(
        self,
        name: str | None = None,
    ) -> None:
        """Initializes the context.

        The function allocates the SDFG and initializes the members properly.

        Args:
            name:       Name of the SDFG object.

        Notes:
            A user should never need to call this function.
        """
        if isinstance(name, str) and not util.VALID_SDFG_OBJ_NAME.fullmatch(name):
            raise ValueError(f"'{name}' is not a valid SDFG name.")

        self.sdfg = dace.SDFG(name=(name or f"unnamed_SDFG_{id(self)}"))
        self.inp_names = ()
        self.out_names = ()
        self.is_finalized = False
        self.start_state = self.sdfg.add_state(label="initial_state", is_start_block=True)
        self.terminal_state = self.start_state

    def validate(self) -> bool:
        """Validate the underlying SDFG.

        The actual SDFG is only validated for finalized SDFGs.
        """
        if len(self.inp_names) == 0:
            raise dace.sdfg.InvalidSDFGError(
                "There are no input arguments.",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if len(self.out_names) == 0:
            raise dace.sdfg.InvalidSDFGError(
                "There are no output arguments.",
                self.sdfg,
                self.sdfg.node_id(self.start_state),
            )
        if self.start_state and (self.start_state is not self.sdfg.start_block):
            raise dace.sdfg.InvalidSDFGError(
                f"Expected to find '{self.start_state}' ({self.sdfg.node_id(self.start_state)}),"
                f" instead found '{self.sdfg.start_block} ({self.sdfg.node_id(self.sdfg.start_block)}).",
                self.sdfg,
                self.sdfg.node_id(self.start_state),
            )
        if self.start_state and ({self.terminal_state} != set(self.sdfg.sink_nodes())):
            raise dace.sdfg.InvalidSDFGError(
                f"Expected to find '{self.terminal_state}' ({self.sdfg.node_id(self.terminal_state)}),"
                f" instead found '{self.sdfg.sink_nodes()}.",
                self.sdfg,
                self.sdfg.node_id(self.terminal_state),
            )
        if not self.is_finalized:
            return True  # More we can not do for an unfinalized SDFG.
        self.sdfg.validate()
        return True
