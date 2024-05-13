# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global utility package for the jax to dace translator."""

from __future__ import annotations

from .debug import _jace_run, compile_jax_sdfg, run_jax_sdfg
from .jax_helper import (
    JaCeVar,
    _propose_jax_name,
    get_jax_var_dtype,
    get_jax_var_name,
    get_jax_var_shape,
    is_tracing_ongoing,
    translate_dtype,
)
from .traits import (
    is_array,
    is_drop_var,
    is_fully_addressable,
    is_jaceified,
    is_jax_array,
    is_jaxified,
    is_non_string_iterable,
    is_on_device,
)
from .util import (
    VALID_JAX_VAR_NAME,
    VALID_SDFG_OBJ_NAME,
    VALID_SDFG_VAR_NAME,
    as_sequence,
)


__all__ = [
    "JaCeVar",
    "as_sequence",
    "compile_jax_sdfg",
    "dataclass_with_default_init",
    "is_array",
    "is_drop_var",
    "is_tracing_ongoing",
    "is_jaceified",
    "is_jaxified",
    "is_jax_array",
    "is_fully_addressable",
    "is_non_string_iterable",
    "is_on_device",
    "get_jax_var_name",
    "get_jax_var_shape",
    "get_jax_var_dtype",
    "translate_dtype",
    "run_jax_sdfg",
    "_jace_run",
    "_propose_jax_name",
    "VALID_JAX_VAR_NAME",
    "VALID_SDFG_OBJ_NAME",
    "VALID_SDFG_VAR_NAME",
]
