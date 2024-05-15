# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dace
from jax import core as jax_core

from jace import util


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
    - `is_finalized` a bool that indicates if `self` represents a finalized or canonical SDFG, see below.
    - `rev_idx` the revision index, used for name mangling, however, outside of a translation process,
        the value carries no meaning.

    Note, that it might happen that a name appears in both the `inp_names` and `out_names` lists.
    This happens if an argument is used both as input and output, and it is not an error.
    In Jax this is called argument donation.

    If the flag `is_finalized` is `True` `self` carries a so called finalized SDFG.
    In this case only the `sdfg`, `inp_names`, `out_names` and `is_finalized` fields remain allocated, all others are set to `None`.
    Furthermore the SDFG is in the so called finalized form which is:
    - All input an output arrays are marked as global.
    - However, there are no `__return` arrays, i.e. all arguments are passed as arguments.
    - Its `arg_names` are set with set `inp_names + out_names`, however,
        arguments that are input and outputs are only listed as inputs.
    """

    sdfg: dace.SDFG
    inp_names: tuple[str, ...]
    out_names: tuple[str, ...]
    jax_name_map: dict[jax_core.Var | util.JaCeVar, str] | None
    start_state: dace.SDFGState | None
    terminal_state: dace.SDFGState | None
    rev_idx: int | None
    is_finalized: bool

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

        self.sdfg = dace.SDFG(name=(name or f"unnamed_SDFG_{id(self)}"))
        self.start_state = self.sdfg.add_state(label="initial_state", is_start_block=True)
        self.terminal_state = self.start_state
        self.jax_name_map = {}
        self.inp_names = ()
        self.out_names = ()
        self.rev_idx = rev_idx
        self.is_finalized = False

    def validate(self) -> bool:
        """Validate the underlying SDFG.

        Only a finalized SDFG can be validated.
        """
        if not self.is_finalized:
            raise dace.sdfg.InvalidSDFGError(
                "SDFG is not finalized.",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
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
        self.sdfg.validate()
        return True
