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

from jax import core as jax_core

from jace import util


class NonStringIterable(Iterable): ...


def is_non_string_iterable(val: Any) -> TypeGuard[NonStringIterable]:
    return isinstance(val, Iterable) and not isinstance(val, str)


def is_jaceified(obj: Any) -> bool:
    """Tests if `obj` is decorated by JaCe.

    Similar to `jace.util.is_jaxified`, but for JaCe object.
    """
    from jace import jax as jjax

    if util.is_jaxified(obj):
        return False
    # Currently it is quite simple because we can just check if `obj`
    #  is derived from `jace.jax.JaceWrapped`, might become harder in the future.
    return isinstance(obj, jjax.JaceWrapped)


def is_drop_var(jax_var: jax_core.Atom | util.JaCeVar) -> bool:
    """Tests if `jax_var` is a drop variable."""

    if isinstance(jax_var, jax_core.DropVar):
        return True
    if isinstance(jax_var, util.JaCeVar):
        return jax_var.name == "_"
    return False


def is_jaxified(obj: Any) -> bool:
    """Tests if `obj` is a "jaxified" object.

    A "jexified" object is an object that was processed by Jax.
    While a return value of `True` guarantees a jaxified object,
    `False` might not proof the contrary.
    """
    import jaxlib
    from jax import _src as jax_src

    # These are all types we consider as jaxify
    jaxifyed_types = (
        jax_core.Primitive,
        # jstage.Wrapped is not runtime chakable
        jax_src.pjit.JitWrapped,
        jaxlib.xla_extension.PjitFunction,
    )
    return isinstance(obj, jaxifyed_types)
