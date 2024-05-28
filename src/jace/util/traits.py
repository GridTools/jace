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
from jax import _src as jax_src, core as jax_core
from jaxlib import xla_extension as jax_xe

import jace.util as util
from jace import stages


def is_jaceified(obj: Any) -> TypeGuard[stages.JaceWrapped]:
    """Tests if `obj` is decorated by JaCe.

    Similar to `is_jaxified`, but for JaCe object.
    """
    if util.is_jaxified(obj):
        return False
    return isinstance(obj, stages.JaceWrapped)


def is_drop_var(jax_var: jax_core.Atom | util.JaCeVar) -> TypeGuard[jax_core.DropVarp]:
    """Tests if `jax_var` is a drop variable, i.e. a variable that is not read from in a Jaxpr."""

    if isinstance(jax_var, jax_core.DropVar):
        return True
    if isinstance(jax_var, util.JaCeVar):
        return jax_var.name == "_" if jax_var.name else False
    return False


def is_jaxified(
    obj: Any,
) -> TypeGuard[jax_core.Primitive | jax_src.pjit.JitWrapped | jax_xe.PjitFunction]:
    """Tests if `obj` is a "jaxified" object.

    A "jaxified" object is an object that was processed by Jax.
    While a return value of `True` guarantees a jaxified object, `False` does not proof the
    contrary. See also `jace.util.is_jaceified()` to tests if something is a Jace object.
    """
    jaxifyed_types = (
        jax_core.Primitive,
        # jax_core.stage.Wrapped is not runtime chakable
        jax_src.pjit.JitWrapped,
        jax_xe.PjitFunction,
    )
    return isinstance(obj, jaxifyed_types)


def is_jax_array(
    obj: Any,
) -> TypeGuard[jax.Array]:
    """Tests if `obj` is a jax array.

    Notes jax array are special as you can not write to them directly.
    Furthermore, they always allocate also on GPU, beside the CPU allocation.
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

    Jax arrays are always on the CPU and GPU (if there is one). Thus for Jax arrays this
    function is more of a test, if there is a GPU or not.
    """
    if is_jax_array(obj):
        try:
            _ = obj.__cuda_array_interface__
            return True
        except AttributeError:
            return False
    return dace.is_gpu_array(obj)


def is_fully_addressable(
    obj: Any,
) -> bool:
    """Tests if `obj` is fully addressable, i.e. is only on this host.

    Notes:
        This function currently assumes that everything that is not a Jax array is always fully
        addressable.
    """
    if is_jax_array(obj):
        return obj.is_fully_addressable
    return True
