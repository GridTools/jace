# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar, cast, overload


_T = TypeVar("_T")


@overload
def as_sequence(value: str) -> Iterable[str]: ...


@overload
def as_sequence(value: Iterable[_T]) -> Iterable[_T]: ...


@overload
def as_sequence(value: _T) -> Iterable[_T]: ...


def as_sequence(value: _T | Iterable[_T]) -> Iterable[_T]:
    from jace.util.traits import is_non_string_iterable

    if is_non_string_iterable(value):
        return value
    return cast(Iterable[_T], [value])


def is_jaceified(obj: Any) -> bool:
    """Tests if `obj` is decorated by JaCe.

    Similar to `jace.util.is_jaxified`, but for JaCe object.
    """
    from jace import jax as jjax, util as jutil

    if jutil.is_jaxified(obj):
        return False
    # Currently it is quite simple because we can just check if `obj`
    #  is derived from `jace.jax.JitWrapped`, might become harder in the future.
    return isinstance(obj, jjax.JitWrapped)
