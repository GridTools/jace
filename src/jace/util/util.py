# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeVar, cast, overload

import jace.util.traits as traits


_T = TypeVar("_T")


@overload
def as_sequence(value: str) -> Iterable[str]: ...


@overload
def as_sequence(value: Iterable[_T]) -> Iterable[_T]: ...


@overload
def as_sequence(value: _T) -> Iterable[_T]: ...


def as_sequence(value: _T | Iterable[_T]) -> Iterable[_T]:
    if traits.is_non_string_iterable(value):
        return value
    return cast(Iterable[_T], [value])


# Valid name for a jax variable.
VALID_JAX_VAR_NAME: re.Pattern = re.compile("(jax[0-9]+_?)|([a-z]+_?)")

# Valid name for an SDFG variable.
VALID_SDFG_VAR_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

# Valid name for an SDFG itself, includes `SDFGState` objects.
VALID_SDFG_OBJ_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")
