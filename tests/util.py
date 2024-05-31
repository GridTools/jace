# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for the testing infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "mkarray",
]


def mkarray(
    shape: Sequence[int] | int,
    dtype: type = np.float64,
) -> np.ndarray:
    """Generates a NumPy ndarray with shape `shape`.

    The function uses the generator that is managed by the `_reset_random_seed()` fixture.
    Thus inside a function the value will be deterministic.

    Args:
        shape:      The shape to use.
        dtype:      The data type to use.
    """
    if isinstance(shape, int):
        shape = (shape,)
    assert shape
    return np.array(np.random.random(shape), dtype=dtype)  # noqa: NPY002
