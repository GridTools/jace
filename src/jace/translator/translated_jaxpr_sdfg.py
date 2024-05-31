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
    """Encapsulates the result of a translation run of the `JaxprTranslationBuilder` object.

    The only valid way to obtain a `TranslatedJaxprSDFG` is by passing a `TranslationContext`,
    that was in turn constructed by `JaxprTranslationBuilder.translate_jaxpr()`, to the
    `postprocess_jaxpr_sdfg()` function.
    This class encapsulates a translated SDFG as well as the meta data needed to compile and run it.

    Contrary to the SDFG that is encapsulated inside the `TranslationContext` object, `self`
    carries a proper SDFG, however:
    - it does not have `__return*` variables, instead all return arguments are passed by arguments,
    - its `arg_names` is set to  `inp_names + out_names`, but arguments that are input and outputs
        are only listed as inputs.

    Attributes:
        sdfg:           The encapsulated SDFG object.
        inp_names:      A list of the SDFG variables that are used as input
        out_names:      A list of the SDFG variables that are used as output.

    The `inp_names` and `out_name` are in the same order as in the original Jaxpr object.
    It might happen that a name appears in both the `inp_names` and `out_names` lists. This happens
    if an argument is used both as input and output, and it is not an error. In Jax this is called
    argument donation.

    Args:
        name:   The name that should be given to the SDFG, optional.
    """

    sdfg: dace.SDFG
    inp_names: tuple[str, ...]
    out_names: tuple[str, ...]

    def __init__(
        self,
        name: str | None = None,
    ) -> None:
        if isinstance(name, str) and not util.VALID_SDFG_OBJ_NAME.fullmatch(name):
            raise ValueError(f"'{name}' is not a valid SDFG name.")

        self.sdfg = dace.SDFG(name=(name or f"unnamed_SDFG_{id(self)}"))
        self.inp_names = ()
        self.out_names = ()

    def validate(self) -> bool:
        """Validate the underlying SDFG."""
        if not self.inp_names:
            raise dace.sdfg.InvalidSDFGError(
                "There are no input arguments.",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if not all(not self.sdfg.arrays[inp].transient for inp in self.inp_names):
            raise dace.sdfg.InvalidSDFGError(
                f"Found transient inputs: {(inp for inp in self.inp_names if self.sdfg.arrays[inp].transient)}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if not self.out_names:
            raise dace.sdfg.InvalidSDFGError(
                "There are no output arguments.",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if not all(not self.sdfg.arrays[out].transient for out in self.out_names):
            raise dace.sdfg.InvalidSDFGError(
                f"Found transient outputs: {(out for out in self.out_names if self.sdfg.arrays[out].transient)}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        self.sdfg.validate()
        return True
