# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Container for storing a translated SDFG."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import dace


if TYPE_CHECKING:
    from jax import tree_util as jax_tree


@dataclasses.dataclass(kw_only=True, frozen=True)
class TranslatedJaxprSDFG:
    """
    Encapsulates a translated SDFG with additional the metadata.

    Contrary to the SDFG that is encapsulated inside an `TranslationContext`
    object, `self` carries a proper SDFG, however:
    - It does not have `__return*` variables, instead all return arguments are
        passed by arguments.
    - All input arguments are passed through arguments mentioned in `inp_names`,
        while the outputs are passed through `out_names`.
    - Only variables listed as in/outputs are non transient.
    - The order inside `inp_names` and `out_names` is the same as in the original Jaxpr.
    - If an input is used as outputs it appears in both `inp_names` and `out_names`.
    - Its `arg_names` is set to  `inp_names + out_names`, but arguments that are
        input and outputs are only listed as inputs.

    The only valid way to obtain a `TranslatedJaxprSDFG` is by passing a
    `TranslationContext`, that was in turn constructed by
    `JaxprTranslationBuilder.translate_jaxpr()`, to the
    `finalize_translation_context()` or preferably to the `postprocess_jaxpr_sdfg()`
    function.

    Attributes:
        sdfg: The encapsulated SDFG object.
        inp_names: Names of the SDFG variables used as inputs.
        out_names: Names of the SDFG variables used as outputs.
        outtree: A pytree describing how to unflatten the output.

    Todo:
        After the SDFG is compiled a lot of code looks strange, because there is
        no container to store the compiled SDFG and the metadata. This class
        should be extended to address this need.
    """

    sdfg: dace.SDFG
    inp_names: tuple[str, ...]
    out_names: tuple[str, ...]
    outtree: jax_tree.PyTreeDef

    def validate(self) -> bool:
        """Validate the underlying SDFG."""
        if any(self.sdfg.arrays[inp].transient for inp in self.inp_names):
            raise dace.sdfg.InvalidSDFGError(
                f"Found transient inputs: {(inp for inp in self.inp_names if self.sdfg.arrays[inp].transient)}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if any(self.sdfg.arrays[out].transient for out in self.out_names):
            raise dace.sdfg.InvalidSDFGError(
                f"Found transient outputs: {(out for out in self.out_names if self.sdfg.arrays[out].transient)}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if self.sdfg.free_symbols:  # This is a simplification that makes our life simple.
            raise dace.sdfg.InvalidSDFGError(
                f"Found free symbols: {self.sdfg.free_symbols}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        self.sdfg.validate()
        return True
