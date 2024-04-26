# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Python library for translating Jax programs into SDFG."""

from __future__ import annotations

from .__about__ import __author__, __copyright__, __license__, __version__, __version_info__
from .jax import grad, jacfwd, jacrev, jit


__all__ = [
    "__author__",
    "__copyright__",
    "grad",
    "jit",
    "jacfwd",
    "jacrev",
    "__license__",
    "__version__",
    "__version_info__",
]
