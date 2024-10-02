# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global utility package for the jax to dace translator."""

from __future__ import annotations

from .definitions import FORBIDDEN_SDFG_VAR_NAMES, VALID_SDFG_OBJ_NAME, VALID_SDFG_VAR_NAME
from .jax_helper import (
    JaCeVar,
    get_jax_literal_value,
    get_jax_var_dtype,
    get_jax_var_name,
    get_jax_var_shape,
    is_tracing_ongoing,
    parse_backend_jit_option,
    propose_jax_name,
    translate_dtype,
)
from .traits import (
    get_strides_for_dace,
    is_array,
    is_c_contiguous,
    is_drop_var,
    is_fully_addressable,
    is_jax_array,
    is_on_device,
    is_scalar,
)


__all__ = [
    "FORBIDDEN_SDFG_VAR_NAMES",
    "VALID_SDFG_OBJ_NAME",
    "VALID_SDFG_VAR_NAME",
    "JaCeVar",
    "get_jax_literal_value",
    "get_jax_var_dtype",
    "get_jax_var_name",
    "get_jax_var_shape",
    "get_strides_for_dace",
    "is_array",
    "is_c_contiguous",
    "is_drop_var",
    "is_fully_addressable",
    "is_jax_array",
    "is_on_device",
    "is_scalar",
    "is_tracing_ongoing",
    "parse_backend_jit_option",
    "propose_jax_name",
    "translate_dtype",
]
