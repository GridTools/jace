# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the broadcast in dim translator.

Todo:
    - `np.meshgrid`
    - `np.expand_dims`
    - `np.ix_`
    - `np.indices`
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jax import numpy as jnp

import jace


def test_bid_scalar():
    """Broadcast a scalar to a matrix."""

    def testee(A: float) -> np.ndarray:
        return jnp.broadcast_to(A, (2, 2))

    for a in [1, 1.0, 3.1415]:
        ref = testee(a)
        res = jace.jit(testee)(a)

        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        assert np.all(res == ref), f"Expected '{ref.tolist()}' got '{res.tolist()}'."


def test_bid_literal():
    """Broadcast a literal to a matrix."""

    def testee(a: float) -> np.ndarray:
        return jnp.broadcast_to(1.0, (10, 10)) + a

    for a in [1, 1.0, 3.1415]:
        ref = testee(a)
        res = jace.jit(testee)(a)
        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        assert np.all(res == ref)


def _expand_dims_test_impl(
    shape: Sequence[int],
    axes: Sequence[int | Sequence[int]],
) -> None:
    """Implementation of the test for `expand_dims()`.

    Args:
        shape:  Shape of the input array.
        axes:   A series of axis that should be tried.
    """
    A = np.random.random(shape)  # noqa: NPY002
    for axis in axes:

        def testee(A):
            return jnp.expand_dims(A, axis)  # noqa: B023  # Binding loop variable.

        ref = testee(A)
        res = jace.jit(testee)(A)

        assert ref.shape == res.shape, f"A.shape = {shape}; Expected: {ref.shape}; Got: {res.shape}"
        assert np.all(ref == res), f"Value error for shape '{shape}' and axis={axis}"


def test_expand_dims():
    """Test various calls to `np.expand_dims()`."""
    _expand_dims_test_impl((10,), [0, -1, 1])
    _expand_dims_test_impl(
        (2, 3, 4, 5),
        [
            0,
            -1,
            (1, 2, 3),
            (3, 2, 1),
        ],
    )
