# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains all traits function needed inside JaCe."""

from __future__ import annotations

from typing import Any, TypeGuard

import dace
import jax
import numpy as np
from jax import core as jax_core

import jace.util as util


def is_drop_var(
    jax_var: jax_core.Atom | util.JaCeVar,
) -> TypeGuard[jax_core.DropVar]:
    """Tests if `jax_var` is a drop variable, i.e. a variable that is not read from in a Jaxpr."""

    if isinstance(jax_var, jax_core.DropVar):
        return True
    if isinstance(jax_var, util.JaCeVar):
        return jax_var.name == "_" if jax_var.name else False
    return False


def is_jax_array(
    obj: Any,
) -> TypeGuard[jax.Array]:
    """Tests if `obj` is a Jax array.

    Note:
     Jax arrays are special as they can not be mutated. Furthermore, they always
     allocate on the CPU _and_ on the GPU, if present.
    """
    return isinstance(obj, jax.Array)


def is_array(
    obj: Any,
) -> bool:
    """Identifies arrays, this also includes Jax arrays."""
    return dace.is_array(obj) or is_jax_array(obj)


def is_scalar(
    obj: Any,
) -> bool:
    """Tests if `obj` is a scalar."""
    # These are the type known to DaCe; Taken from `dace.dtypes`.
    known_types = {
        bool,
        int,
        float,
        complex,
        np.intc,
        np.uintc,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.longlong,
        np.ulonglong,
    }
    return type(obj) in known_types


def is_on_device(
    obj: Any,
) -> bool:
    """Tests if `obj` is on a device.

    Jax arrays are always on the CPU and GPU (if there is one). Thus for Jax
    arrays this function is more of a test, if there is a GPU at all.
    """
    if is_jax_array(obj):
        return hasattr(obj, "__cuda_array_interface__")
    return dace.is_gpu_array(obj)


def is_fully_addressable(
    obj: Any,
) -> bool:
    """Tests if `obj` is fully addressable, i.e. is only on this host."""
    if is_jax_array(obj):
        return obj.is_fully_addressable
    return True
