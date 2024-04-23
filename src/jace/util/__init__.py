# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global utility package for the jax to dace translator."""

from __future__ import annotations

from .jax import JaCeVar, get_jax_var_dtype, get_jax_var_name, get_jax_var_shape, translate_dtype
from .util import ensure_iterability


__all__ = [
    "get_jax_var_name",
    "get_jax_var_shape",
    "get_jax_var_dtype",
    "ensure_iterability",
    "translate_dtype",
    "JaCeVar",
]
