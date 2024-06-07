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


__all__ = ["mkarray"]


def mkarray(shape: Sequence[int] | int, dtype: type = np.float64) -> np.ndarray:
    """Generates a NumPy ndarray with shape `shape`.

    The function uses the generator that is managed by the `_reset_random_seed()` fixture.
    Thus inside a function the value will be deterministic.

    Args:
        shape:      The shape to use.
        dtype:      The data type to use.

    Notes:
        Floating point based values are generated in the range 0 to 1.0.
    """

    if shape == ():
        return mkarray((1,), dtype)[0]
    if isinstance(shape, int):
        shape = (shape,)

    if dtype == np.bool_:
        return np.random.random(shape) > 0.5  # noqa: NPY002
    if np.issubdtype(dtype, np.integer):
        iinfo: np.iinfo = np.iinfo(dtype)
        return np.random.randint(low=iinfo.min, high=iinfo.max, size=shape, dtype=dtype)  # noqa: NPY002
    if np.issubdtype(dtype, np.complexfloating):
        return np.array(mkarray(shape, np.float64) + 1.0j * mkarray(shape, np.float64), dtype=dtype)
    return np.array(np.random.random(shape), dtype=dtype)  # noqa: NPY002
