# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains all general helper functions needed inside the translator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def list_to_dict(inp: Sequence[tuple[None | Any, Any]]) -> dict[Any, Any]:
    """This method turns a `list` of pairs into a `dict` and applies a `None` filter.

    The function will only include pairs whose key, i.e. first element is not `None`.
    """
    return {k: v for k, v in inp if k is not None}
