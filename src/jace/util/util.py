# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def ensure_iterability(
    x: Any,
    ign_str: bool = True,
) -> Iterable[Any]:
    """Ensures that `x` is iterable.

    By default strings are _not_ considered iterable.

    Args:
        x:          To test.
        ign_str:    Ignore that a string is iterabile.
    """
    if ign_str and isinstance(x, str):
        x = [x]
    elif isinstance(x, Iterable):
        pass
    return x


def list_to_dict(inp: Sequence[tuple[None | Any, Any]]) -> dict[Any, Any]:
    """This method turns a `list` of pairs into a `dict` and applies a `None` filter.

    The function will only include pairs whose key, i.e. first element is not `None`.
    """
    return {k: v for k, v in inp if k is not None}
