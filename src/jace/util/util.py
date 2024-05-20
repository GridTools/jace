# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final, TypeVar, cast, overload

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


# Valid name for an SDFG variable.
VALID_SDFG_VAR_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

# Valid name for an SDFG itself, includes `SDFGState` objects.
VALID_SDFG_OBJ_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")


# fmt: off
# This is a set of all names that are invalid SDFG names.
FORBIDDEN_SDFG_VAR_NAMES: Final[set[str]] = {
    # These should be most of the C++ keywords, it is more important to have the short ones.
    #  Taken from 'https://learn.microsoft.com/en-us/cpp/cpp/keywords-cpp?view=msvc-170'
    "alignas", "alignof", "and", "asm", "auto", "bitand", "bitor", "bool", "break", "case",
    "catch", "char", "class", "compl", "concept", "const", "consteval", "constexpr",
    "constinit", "continue", "decltype", "default", "delete", "directive", "do", "double",
    "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend",
    "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not",
    "nullptr", "operator", "or", "private", "protected", "public", "register", "requires",
    "return", "short", "signed", "sizeof", "static", "struct", "switch", "template", "this",
    "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "while", "xor", "std",  "",
}
# fmt: on
