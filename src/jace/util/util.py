# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar, cast, overload


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
