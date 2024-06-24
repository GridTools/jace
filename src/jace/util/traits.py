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

from jace import util


def is_drop_var(jax_var: jax_core.Atom | util.JaCeVar) -> TypeGuard[jax_core.DropVar]:
    """Tests if `jax_var` is a drop variable."""
    if isinstance(jax_var, jax_core.DropVar):
        return True
    if isinstance(jax_var, util.JaCeVar):
        return jax_var.name == "_" if jax_var.name else False
    return False


def is_jax_array(obj: Any) -> TypeGuard[jax.Array]:
    """
    Tests if `obj` is a JAX array.

    Note:
        JAX arrays are special as they can not be mutated. Furthermore, they always
        allocate on the CPU _and_ on the GPU, if present.
    """
    return isinstance(obj, jax.Array)


def is_array(obj: Any) -> TypeGuard[jax.Array]:
    """Identifies arrays, this also includes JAX arrays."""
    # `dace.is_array()` does not seem to recognise shape zero arrays.
    return isinstance(obj, np.ndarray) or dace.is_array(obj) or is_jax_array(obj)


def is_scalar(obj: Any) -> bool:
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


def get_strides_for_dace(obj: Any) -> tuple[int, ...] | None:
    """
    Get the strides of `obj` in a DaCe compatible format.

    The function returns the strides in number of elements, as it is used inside
    DaCe and not in bytes as it is inside NumPy. As in NumPy and DaCe the function
    returns `None` to indicate standard C order.

    Note:
        If `obj` is not array like an error is generated.
    """
    if not is_array(obj):
        raise TypeError(f"Passed '{obj}' ({type(obj).__name__}) is not array like.")

    if is_jax_array(obj):
        if not is_fully_addressable(obj):
            raise NotImplementedError("Sharded jax arrays are not supported.")
        obj = obj.__array__()
    assert hasattr(obj, "strides")

    if obj.strides is None:
        return None
    if not hasattr(obj, "itemsize"):
        # No `itemsize` member so we assume that it is already in elements.
        return obj.strides

    return tuple(stride // obj.itemsize for stride in obj.strides)


def is_on_device(obj: Any) -> bool:
    """
    Tests if `obj` is on a device.

    JAX arrays are always on the CPU and GPU (if there is one). Thus for JAX
    arrays this function is more of a test, if there is a GPU at all.
    """
    if is_jax_array(obj):
        return hasattr(obj, "__cuda_array_interface__")
    return dace.is_gpu_array(obj)


def is_fully_addressable(obj: Any) -> bool:
    """Tests if `obj` is fully addressable, i.e. is only on this host."""
    if is_jax_array(obj):
        return obj.is_fully_addressable
    return True


def is_c_contiguous(obj: Any) -> bool:
    """Tests if `obj` is in C order."""
    if not is_array(obj):
        return False
    if is_jax_array(obj):
        obj = obj.__array__()
    return obj.flags["C_CONTIGUOUS"]
