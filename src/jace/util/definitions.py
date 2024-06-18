# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions of patterns for valid names."""

from __future__ import annotations

import re
from typing import Final


#: Valid name for an SDFG variable.
VALID_SDFG_VAR_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

#: Valid name for an SDFG itself, includes `SDFGState` objects.
VALID_SDFG_OBJ_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")


# fmt: off
#: This is a set of all names that are invalid SDFG names.
FORBIDDEN_SDFG_VAR_NAMES: Final[set[str]] = {
    # These should be most of the C++ keywords, it is more important to have the short
    #  ones. Taken from 'https://learn.microsoft.com/en-us/cpp/cpp/keywords-cpp?view=msvc-170'
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
