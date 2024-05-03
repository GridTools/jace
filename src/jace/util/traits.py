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


def is_drop_var(jax_var: jax_core.Atom | util.JaCeVar) -> bool:
    """Tests if `jax_var` is a drop variable."""

    if isinstance(jax_var, jax_core.DropVar):
        return True
    if isinstance(jax_var, util.JaCeVar):
        return jax_var.name == "_"
    return False
