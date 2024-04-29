# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global utility package for the jax to dace translator."""

from __future__ import annotations

from .debug import _jace_run, run_memento
from .jax import JaCeVar, get_jax_var_dtype, get_jax_var_name, get_jax_var_shape, translate_dtype
from .revision_counter import RevisionCounterManager
from .util import ensure_iterability, list_to_dict


__all__ = [
    "RevisionCounterManager",
    "JaCeVar",
    "get_jax_var_name",
    "get_jax_var_shape",
    "get_jax_var_dtype",
    "ensure_iterability",
    "translate_dtype",
    "list_to_dict",
    "run_memento",
    "_jace_run",
]
