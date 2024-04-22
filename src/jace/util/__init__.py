# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global utility package for the jax to dace translator."""

from __future__ import annotations

from .jax import get_jax_var_name, translate_dtype
from .util import ensure_iterability


__all__ = [
    "get_jax_var_name",
    "ensure_iterability",
    "translate_dtype",
]
