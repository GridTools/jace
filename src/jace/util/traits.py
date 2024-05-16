# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains all traits function needed inside JaCe."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeGuard

import dace
from jax import _src as jax_src, core as jax_core
from jaxlib import xla_extension as jax_xe

from jace import jax as jjax, util


class NonStringIterable(Iterable): ...


def is_non_string_iterable(val: Any) -> TypeGuard[NonStringIterable]:
    return isinstance(val, Iterable) and not isinstance(val, str)


def is_jaceified(obj: Any) -> TypeGuard[jjax.JaceWrapped]:
    """Tests if `obj` is decorated by JaCe.

    Similar to `jace.util.is_jaxified`, but for JaCe object.
    """
    if util.is_jaxified(obj):
        return False
    # Currently it is quite simple because we can just check if `obj`
    #  is derived from `jace.jax.JaceWrapped`, might become harder in the future.
    return isinstance(obj, jjax.JaceWrapped)


def is_drop_var(jax_var: jax_core.Atom | util.JaCeVar) -> TypeGuard[jax_core.DropVarp]:
    """Tests if `jax_var` is a drop variable, i.e. a variable that is not read from in a Jaxpr."""

    if isinstance(jax_var, jax_core.DropVar):
        return True
    if isinstance(jax_var, util.JaCeVar):
        # We type narrow it to a pure jax DropVar, because essentially
        #  you can not do anything with it.
        return jax_var.name == "_"
    return False


def is_jaxified(
    obj: Any,
) -> TypeGuard[jax_core.Primitive | jax_src.pjit.JitWrapped | jax_xe.PjitFunction]:
    """Tests if `obj` is a "jaxified" object.

    A "jaxified" object is an object that was processed by Jax.
    While a return value of `True` guarantees a jaxified object, `False` might not proof the contrary.
    See also `jace.util.is_jaceified()` to tests if something is a Jace object.
    """

    # These are all types we consider as jaxify
    jaxifyed_types = (
        jax_core.Primitive,
        # jstage.Wrapped is not runtime chakable
        jax_src.pjit.JitWrapped,
        jax_xe.PjitFunction,
    )
    return isinstance(obj, jaxifyed_types)


def is_jax_array(
    obj: Any,
) -> bool:
    """Tests if `obj` is a jax array.

    Todo:
        Find the Jax type for `TypeGuard`.
    """
    # Currently this seams to be the besst way to identify Jax arrays.
    return all(hasattr(obj, x) for x in ["sharding", "is_fully_addressable"])


def is_array(
    obj: Any,
) -> bool:
    """Identifies arrays, this also includes Jax arrays."""
    return is_jax_array(obj) or dace.is_array(obj)


def is_on_device(
    obj: Any,
) -> bool:
    """Tests if `obj` is on a device."""
    # The problem is, that we can not test if `__cuda_array_interface__` exists.
    #  because Jax array have that even on CPU, thus it is a bit mnore complex.
    if is_jax_array(obj):
        obj = obj.__array__(copy=False)
    return dace.is_gpu_array(obj)


def is_fully_addressable(
    obj: Any,
) -> bool:
    """Tests if `obj` is fully addreassable, i.e. is only on this host.

    Notes:
        The function (currently) assumes that everything that is not a distributed
            Jax array is on this host.
    """
    if is_jax_array(obj):
        return obj.is_fully_addressable()
    return True
