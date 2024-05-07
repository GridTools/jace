# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reimplementation of the `jax.stages` module.

The module either imports or reimplements Jax classes.
In case classes/functions are reimplemented they might be slightly different to fit their usage within Jace.

As in Jax Jace has different stages, the terminology is taken from [Jax' AOT-Tutorial](https://jax.readthedocs.io/en/latest/aot.html).
- Stage out:
    In this phase we translate an executable python function into Jaxpr.
- Lower:
    This will transform the Jaxpr into an SDFG equivalent.
    As a implementation note, currently this and the previous step are handled as a single step.
- Compile:
    This will turn the SDFG into an executable object, see `dace.codegen.CompiledSDFG`.
- Execution:
    This is the actual running of the computation.
"""

from __future__ import annotations

from jax.stages import CompilerOptions

from .a_stage import Stage
from .jace_compiled import JaceCompiled
from .jace_lowered import JaceLowered
from .jace_wrapped import JaceWrapped


__all__ = [
    "Stage",
    "CompilerOptions",
    "JaceWrapped",
    "JaceLowered",
    "JaceCompiled",
]
