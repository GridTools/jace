# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global utility package for the jax to dace translator."""

from __future__ import annotations

from .compiling import (
    compile_jax_sdfg,
    run_jax_sdfg,
)
from .jax_helper import (
    JaCeVar,
    get_jax_var_dtype,
    get_jax_var_name,
    get_jax_var_shape,
    is_tracing_ongoing,
    propose_jax_name,
    translate_dtype,
)
from .traits import (
    is_array,
    is_drop_var,
    is_fully_addressable,
    is_jaceified,
    is_jax_array,
    is_jaxified,
    is_on_device,
    is_scalar,
)
from .util import (
    FORBIDDEN_SDFG_VAR_NAMES,
    VALID_SDFG_OBJ_NAME,
    VALID_SDFG_VAR_NAME,
    as_sequence,
)


__all__ = [
    "VALID_SDFG_OBJ_NAME",
    "VALID_SDFG_VAR_NAME",
    "FORBIDDEN_SDFG_VAR_NAMES",
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
    "is_on_device",
    "is_scalar",
    "get_jax_var_dtype",
    "get_jax_var_name",
    "get_jax_var_shape",
    "translate_dtype",
    "run_jax_sdfg",
    "propose_jax_name",
]
