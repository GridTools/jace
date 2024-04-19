# JaCe - JAX jit using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality to identify types of objects."""

from __future__ import annotations

from typing import Any, Sequence


def is_str(
    *args: Sequence[Any],
    allow_empty: bool = True,
) -> bool:
    """Tests if its arguments are strings.

    By default empty strings are also considered as strings.
    However, by setting 'allow_empty' to 'False' the function will consider them not as string.
    In case no arguments were passed to the function 'False' will be returned.
    """
    if len(args) == 0:
        return False

    elif allow_empty:
        for x in args:
            if not isinstance(x, str):
                return False  # Not a string
        # end for(x):
    else:
        for x in args:
            if not isinstance(x, str):
                return False  # Not a string
            if len(x) == 0:
                return False  # A string but empty; and check enabled
        # end for(x):
    # end if:

    return True


def is_iterable(
    x: Any,
    ign_str: bool = True,
) -> bool:
    """Test if 'x' is iterable, with an exception for strings.

    By default this function considers strings as not iterable.
    The idea is that a string is in most cases not a collection of individual characters, but should be seen as a whole.
    However, by setting 'ign_str' to 'False' a string is also considered as an iterable.

    Args:
        x:          The object to check.
        ign_str:     Ignore strings, defaults to 'True'.
    """
    from collections.abc import Iterable

    # We do not consider strings as iterable.
    if ign_str and is_str(x):
        return False

    # Based on: https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable/61139278
    return isinstance(x, Iterable)
