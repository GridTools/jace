# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global utility package for the jax to dace translator."""

from __future__ import annotations

from .debug import _jace_run, run_jax_sdfg
from .jax_helper import (
    JaCeVar,
    _propose_jax_name,
    get_jax_var_dtype,
    get_jax_var_name,
    get_jax_var_shape,
    is_tracing_ongoing,
    translate_dtype,
)
from .traits import is_drop_var, is_jaceified, is_jaxified, is_non_string_iterable
from .util import as_sequence


__all__ = [
    "as_sequence",
    "is_drop_var",
    "is_tracing_ongoing",
    "is_jaceified",
    "is_jaxified",
    "is_non_string_iterable",
    "JaCeVar",
    "get_jax_var_name",
    "get_jax_var_shape",
    "get_jax_var_dtype",
    "translate_dtype",
    "run_jax_sdfg",
    "_jace_run",
    "_propose_jax_name",
]
