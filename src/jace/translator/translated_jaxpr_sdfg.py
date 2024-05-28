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

    This class is used by the `JaxprTranslationDriver` to store the context of the SDFG that is
    currently under construction and the return value of `JaxprTranslationDriver.translate_jaxpr()`.
    A user should never create a `TranslatedJaxprSDFG` manually.

    It might happen that a name appears in both the `inp_names` and `out_names` lists. This happens
    if an argument is used both as input and output, and it is not an error. In Jax this is called
    argument donation.

    By default `self` encapsulates a canonical SDFG, see `JaxprTranslationDriver` for more
    information on this. However, if `is_finalized` is set, then `self` contains a finalized SDFG,
    which differs from a canonical SDFG in the following ways:
    - all input and output arrays are marked as global,
    - however, there are no `__return` arrays, i.e. all return values are passed as arguments,
    - its `arg_names` are set with set `inp_names + out_names`, however, arguments that are input
        and outputs are only listed as inputs,
    - only the `sdfg`, `inp_names`, `out_names` and `is_finalized` are guaranteed to be not `None`.

    Attributes:
        sdfg:           The SDFG object that was created.
        inp_names:      A list of the SDFG variables that are used as input, same order as `Jaxpr.invars`.
        out_names:      A list of the SDFG variables that are used as output, same order as `Jaxpr.outvars`.
        start_state:    The first state in the SDFG state machine.
        terminal_state: The (currently) last state in the state machine.
        is_finalized:   Indicates if `self` represents a finalized or canonical SDFG.

    Args:
        name:   The name that should be given to the SDFG, optional.
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
        However, a user should never call this function directly.

        Args:
            name:       Name of the SDFG object.
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
