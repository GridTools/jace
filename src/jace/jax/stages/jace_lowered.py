# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jace.jax.stages.Lowered` stage for Jace."""

from __future__ import annotations

import json
from typing import Any, Final

from jace import optimization, util
from jace.jax import stages
from jace.jax.stages import translation_cache as tcache
from jace.translator import post_translation as ptrans
from jace.util import dace_helper as jdace


class JaceLowered(stages.Stage):
    """Represents the original computation that was lowered to SDFG."""

    _trans_sdfg: ptrans.FinalizedJaxprSDFG

    DEF_COMPILER_OPTIONS: Final[dict[str, Any]] = {
        "auto_opt": True,
        "simplify": True,
    }

    def __init__(
        self,
        trans_sdfg: ptrans.FinalizedJaxprSDFG,
    ) -> None:
        """Constructs the wrapper."""
        if trans_sdfg.inp_names is None:
            raise ValueError("Input names must be defined.")
        if trans_sdfg.out_names is None:
            raise ValueError("Output names must be defined.")
        if trans_sdfg.csdfg is not None:
            raise ValueError("SDFG is already compiled.")
        self._trans_sdfg = trans_sdfg

    @tcache.cached_translation
    def compile(
        self,
        compiler_options: stages.CompilerOptions | None = None,  # Unused arguments
    ) -> stages.JaceCompiled:
        """Compile the SDFG.

        Returns an Object that encapsulates a compiled SDFG object.
        """
        from copy import deepcopy

        # The reason why we have to deepcopy the SDFG
        #  All optimization DaCe functions works in place, if we would not copy the SDFG first, then we would have a problem.
        #  Because, these optimization would then have a feedback of the SDFG object which is stored inside `self`.
        #  Thus if we would run this code `(jaceLoweredObject := jaceWrappedObject.lower()).compile({opti=True})` would return an optimized object.
        #  However, if we would now call `jaceWrappedObject.lower()` (with the same arguments as before, we would get `jaceLoweredObject`,
        #  but it would actually contain an already optimized SDFG, which is not what we want.
        fsdfg: ptrans.FinalizedJaxprSDFG = deepcopy(self._trans_sdfg)
        optimization.jace_auto_optimize(
            fsdfg, **({} if compiler_options is None else compiler_options)
        )
        csdfg: jdace.CompiledSDFG = util.compile_jax_sdfg(fsdfg, cache=False)

        return stages.JaceCompiled(
            csdfg=csdfg,
            inp_names=fsdfg.inp_names,
            out_names=fsdfg.out_names,
        )

    def compiler_ir(self, dialect: str | None = None) -> ptrans.FinalizedJaxprSDFG:
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._trans_sdfg
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def as_html(self, filename: str | None = None) -> None:
        """Runs the `view()` method of the underlying SDFG.

        This is a Jace extension.
        """
        self.compiler_ir().sdfg.view(filename=filename, verbose=False)

    def as_text(self, dialect: str | None = None) -> str:
        """Textual representation of the SDFG.

        By default, the function will return the Json representation of the SDFG.
        However, by specifying `'html'` as `dialect` the function will call `view()` on the underlying SDFG.

        Notes:
            You should prefer `self.as_html()` instead of this function.
        """
        if (dialect is None) or (dialect.upper() == "JSON"):
            return json.dumps(self.compiler_ir().sdfg.to_json())
        if dialect.upper() == "HTML":
            self.as_html()
            return ""  # For the interface
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def cost_analysis(self) -> Any | None:
        """A summary of execution cost estimates.

        Not implemented use the DaCe [instrumentation API](https://spcldace.readthedocs.io/en/latest/optimization/profiling.html) directly.
        """
        raise NotImplementedError()
