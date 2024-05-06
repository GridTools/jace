# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This package mimics parts of the interface of the `jax` package that is supported by JaCe."""

from __future__ import annotations

from .api import grad, jacfwd, jacrev, jit
from .jace_compiled import JaceCompiled
from .jace_jitted import JaceWrapped
from .jace_lowered import JaceLowered
from .stages import (  # type: ignore[attr-defined] # not explicit exported
    Compiled,
    CompilerOptions,
    Lowered,
    Wrapped,
)


__all__ = [
    "Compiled",
    "CompilerOptions",
    "JaceWrapped",
    "JaceLowered",
    "JaceCompiled",
    "Lowered",
    "Wrapped",
    "jit",
    "jacfwd",
    "jacrev",
    "grad",
]
