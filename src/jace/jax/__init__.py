# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This package mimics parts of the interface of the `jax` package that is supported by JaCe."""

from __future__ import annotations

from .api import grad, jacfwd, jacrev, jit
from .api_helper import JitWrapped


__all__ = [
    "JitWrapped",
    "jit",
    "jacfwd",
    "jacrev",
    "grad",
]