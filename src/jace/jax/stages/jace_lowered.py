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

from jace import translator, util
from jace.jax import stages
from jace.util import dace_helper as jdace, translation_cache as tcache


class JaceLowered(stages.Stage):
    """Represents the original computation that was lowered to SDFG."""

    __slots__ = (
        "_translated_sdfg",
        "_cache",
    )

    _translated_sdfg: translator.TranslatedJaxprSDFG
    _cache: tcache.TranslationCache

    DEF_COMPILER_OPTIONS: Final[dict[str, Any]] = {
        "auto_opt": True,
        "simplify": True,
    }

    def __init__(
        self,
        translated_sdfg: translator.TranslatedJaxprSDFG,
    ) -> None:
        """Constructs the wrapper."""
        if translated_sdfg.inp_names is None:
            raise ValueError("Input names must be defined.")
        if translated_sdfg.out_names is None:
            raise ValueError("Output names must be defined.")
        self._translated_sdfg = translated_sdfg
        self._cache: tcache.TranslationCache = tcache.get_cache(self)

    def optimize(
        self,
        **kwargs: Any,  # noqa: ARG002  # Unused argument
    ) -> JaceLowered:
        """Perform optimization _inplace_ and return `self`.

        Notes:
            Currently no optimization is performed.
        """
        # TODO(phimuell): Think really hard what we should do here, to avoid strange behaviour.
        #                   I am not fully sure if we should include the SDFG value in the caching.

        # TODO(phimuell):
        #   - remove the inplace modification.
        #   - Somehow integrate it into the caching strategy.
        #
        #  If we would not integrate it into the caching strategy, then calling `lower()` on
        #  the wrapped object would return the original object, but with a modified, already optimized SDFG.
        return self

    @tcache.cached_translation
    def compile(
        self,
        compiler_options: stages.CompilerOptions | None = None,  # noqa: ARG002  # Unused arguments
    ) -> stages.JaceCompiled:
        """Compile the SDFG.

        Returns an Object that encapsulates a compiled SDFG object.
        """
        csdfg: jdace.CompiledSDFG = util.compile_jax_sdfg(self._translated_sdfg)
        return stages.JaceCompiled(
            csdfg=csdfg,
            inp_names=self._translated_sdfg.inp_names,
            out_names=self._translated_sdfg.out_names,
        )

    def compiler_ir(self, dialect: str | None = None) -> translator.TranslatedJaxprSDFG:
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._translated_sdfg
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
