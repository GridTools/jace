# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def ensure_iterability(
    x: Any,
    dcyp: bool = False,
    scyp: bool = False,
    ign_str: bool = True,
) -> Iterable[Any]:
    """Ensures that 'x' is iterable.

    By default a string is _not_ considered as a sequence of chars but as one object.

    Args:
        x:          To test.
        dcyp:       Perform a deep copy on the reurned object, takes precedence.
        scyp:       Perform a shallow copy on the returned object.
        ign_str:    Ignore that a string is iterabile.
    """
    import copy

    if ign_str and isinstance(x, str):
        x = [x]  # Turn a string into an interable
    elif isinstance(x, Iterable):
        pass  # Already an iterable
    if dcyp:
        x = copy.deepcopy(x)
    elif scyp:
        x = copy.copy(x)
    return x
