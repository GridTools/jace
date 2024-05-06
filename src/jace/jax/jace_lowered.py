# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jace.jax.stages.Lowered` stage for Jace."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from jace import jax as jjax, translator, util
from jace.jax import stages
from jace.util import dace_helper as jdace


class JaceLowered(stages.Lowered):
    """Represents the original computation that was lowered to SDFG."""

    __slots__ = ("_translated_sdfg",)

    _translated_sdfg: translator.TranslatedJaxprSDFG

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

    def compiler_ir(self, dialect: str | None = None) -> translator.TranslatedJaxprSDFG:
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._translated_sdfg
        raise ValueError(f"Unknown dialect '{dialect}'.")

    @override
    def optimize(
        self,
        **kwargs: Any,
    ) -> jjax.JaceLowered:
        """Perform optimization _inplace_ and return `self`.

        Currently no optimization is done, thus `self` is returned unmodified.
        """
        return self

    @override
    def compile(
        self,
        compiler_options: jjax.CompilerOptions | None = None,
    ) -> jjax.JaceCompiled:
        """Compile the SDFG.

        Returns an Object that encapsulates a
        """
        csdfg: jdace.CompiledSDFG = util.compile_jax_sdfg(
            self._translated_sdfg,
            force=True,
            save=False,
        )
        return jjax.JaceCompiled(
            csdfg=csdfg,
            inp_names=self._translated_sdfg.inp_names,  # type: ignore[arg-type]  # init guarantees this
            out_names=self._translated_sdfg.out_names,  # type: ignore[arg-type]
        )
