# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for slicing translator."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jace

from tests import util as testutil


@pytest.fixture()
def A_20x20x20() -> np.ndarray:
    return testutil.make_array((20, 20, 20))


@pytest.fixture()
def A_4x4x4x4() -> np.ndarray:
    return testutil.make_array((4, 4, 4, 4))


@pytest.fixture(
    params=[
        (1, 2, 1, 2),
        (0, 0, 0, 0),
        (3, 3, 3, 3),  # Will lead to readjustment of the start index.
        (3, 1, 3, 0),  # Will lead to readjustment of the start index.
    ]
)
def full_dynamic_start_idx(request) -> tuple[int, int, int, int]:
    """Start indexes for the slice window of `test_dynamic_slice_full_dynamic()`."""
    return request.param


def test_slice_no_strides(A_20x20x20: np.ndarray) -> None:
    """Test without strides."""

    def testee(A: np.ndarray) -> jax.Array:
        # Read as: A[2:18, 3:19, 4:17]
        return jax.lax.slice(A, (2, 3, 4), (18, 19, 17), None)

    ref = testee(A_20x20x20)
    res = jace.jit(testee)(A_20x20x20)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_strides(A_20x20x20: np.ndarray) -> None:
    """Test with strides."""

    def testee(A: np.ndarray) -> jax.Array:
        # Read as: A[2:18:1, 3:19:2, 4:17:3]
        return jax.lax.slice(A, (2, 3, 4), (18, 19, 17), (1, 2, 3))

    ref = testee(A_20x20x20)
    res = jace.jit(testee)(A_20x20x20)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_dynamic_slice_full_dynamic(
    A_4x4x4x4: np.ndarray, full_dynamic_start_idx: tuple[int, int, int, int]
) -> None:
    def testee(A: np.ndarray, s1: int, s2: int, s3: int, s4: int) -> jax.Array:
        return jax.lax.dynamic_slice(A, (s1, s2, s3, s4), (2, 2, 2, 2))

    res = jace.jit(testee)(A_4x4x4x4, *full_dynamic_start_idx)
    ref = testee(A_4x4x4x4, *full_dynamic_start_idx)

    assert np.all(ref == res)


def test_dynamic_slice_partially_dynamic(A_4x4x4x4: np.ndarray) -> None:
    def testee(A: np.ndarray, s1: int, s2: int) -> jax.Array:
        return jax.lax.dynamic_slice(A, (s1, 1, s2, 2), (2, 2, 2, 2))

    res = jace.jit(testee)(A_4x4x4x4, 1, 2)
    ref = testee(A_4x4x4x4, 1, 2)

    assert np.all(ref == res)


def test_dynamic_slice_full_literal(A_4x4x4x4: np.ndarray) -> None:
    def testee(A: np.ndarray) -> jax.Array:
        return jax.lax.dynamic_slice(A, (0, 1, 0, 2), (2, 2, 2, 2))

    res = jace.jit(testee)(A_4x4x4x4)
    ref = testee(A_4x4x4x4)

    assert np.all(ref == res)
