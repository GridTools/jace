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
def A_4x4() -> np.ndarray:
    return testutil.mkarray((4, 4))


@pytest.fixture()
def A_4x4x4x4() -> np.ndarray:
    return testutil.mkarray((4, 4, 4, 4))


@pytest.fixture(
    params=[
        (1, 2, 1, 2),
        (0, 0, 0, 0),
        (3, 3, 3, 3),  # Will lead to readjustment of the start index.
        (3, 1, 3, 0),  # Will lead to readjustment of the start index.
    ]
)
def full_dynamic_start_idx(
    request,
) -> tuple[int, int, int, int]:
    """Start indexes for the slice window of `test_dynamic_slice_full_dynamic()`."""
    return request.param


def test_slice_sub_view(
    A_4x4: np.ndarray,
) -> None:
    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:3, 1:3]

    ref = A_4x4[1:3, 1:3]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_rslice(
    A_4x4: np.ndarray,
) -> None:
    """Only extracting rows."""

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:3]

    ref = A_4x4[1:3]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_cslice(
    A_4x4: np.ndarray,
) -> None:
    """Slicing some columns."""

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        # NOTE: using `A[..., 1:3]` would trigger the `gather` primitive.
        return A[:, 1:3]

    ref = A_4x4[:, 1:3]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_singelton(
    A_4x4: np.ndarray,
) -> None:
    """Only extracting a single value."""

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:2, 1:2]

    ref = A_4x4[1:2, 1:2]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


@pytest.mark.skip(reason="Missing 'gather' translator.")
def test_slice_strides_vec() -> None:
    """Slice with strides.

    Although the translator (and the primitive) would support a stride, for some
    reason Jax makes a `gather` operation out of it, that is not yet supported.
    """

    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:15:2]

    A = np.arange(16)
    ref = testee(A)
    res = jace.jit(testee)(A)

    assert ref.shape == res.shape
    assert np.all(ref == res)


@pytest.mark.skip(reason="Missing 'concatenate' translator.")
def test_slice_strides(
    A_4x4: np.ndarray,
) -> None:
    """Using strides in a 2D matrix.

    See `test_slice_strides_vec()` why the test is skipped.
    """

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[::2, ::2]

    ref = A_4x4[::2, ::2]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_too_big(
    A_4x4: np.ndarray,
) -> None:
    """Tests what happens if we specify a size that is too big."""

    def testee(A: np.ndarray) -> np.ndarray:
        return A[:20]

    ref = testee(A_4x4)
    res = jace.jit(testee)(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_dynamic_slice_full_dynamic(
    A_4x4x4x4: np.ndarray,
    full_dynamic_start_idx: tuple[int, int, int, int],
) -> None:
    def testee(A: np.ndarray, s1: int, s2: int, s3: int, s4: int) -> jax.Array:
        return jax.lax.dynamic_slice(A, (s1, s2, s3, s4), (2, 2, 2, 2))

    res = jace.jit(testee)(A_4x4x4x4, *full_dynamic_start_idx)
    ref = testee(A_4x4x4x4, *full_dynamic_start_idx)

    assert np.all(ref == res)


def test_dynamic_slice_partially_dynamic(
    A_4x4x4x4: np.ndarray,
) -> None:
    def testee(A: np.ndarray, s1: int, s2: int) -> jax.Array:
        return jax.lax.dynamic_slice(A, (s1, 1, s2, 2), (2, 2, 2, 2))

    res = jace.jit(testee)(A_4x4x4x4, 1, 2)
    ref = testee(A_4x4x4x4, 1, 2)

    assert np.all(ref == res)


def test_dynamic_slice_full_literal(
    A_4x4x4x4: np.ndarray,
) -> None:
    def testee(A: np.ndarray) -> jax.Array:
        return jax.lax.dynamic_slice(A, (0, 1, 0, 2), (2, 2, 2, 2))

    res = jace.jit(testee)(A_4x4x4x4)
    ref = testee(A_4x4x4x4)

    assert np.all(ref == res)
