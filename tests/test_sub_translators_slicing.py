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


@pytest.fixture(autouse=True)
def _enable_x64_mode_in_jax():
    """Ensures that x64 mode in Jax ins enabled."""
    with jax.experimental.enable_x64():
        yield


@pytest.fixture()
def A_4x4():
    return np.arange(16).reshape((4, 4))


def test_slice_sub_view(A_4x4):
    """Simple extraction of a subsize."""

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:3, 1:3]

    ref = A_4x4[1:3, 1:3]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_rslice(A_4x4):
    """Only slicing some rows."""

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:3]

    ref = A_4x4[1:3]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_cslice(A_4x4):
    """Slicing some columns."""

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        # NOTE: using `A[..., 1:3]` would trigger the `gather` primitive.
        return A[:, 1:3]

    ref = A_4x4[:, 1:3]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_slice_singelton(A_4x4):
    """Only extracting a single value."""

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:2, 1:2]

    ref = A_4x4[1:2, 1:2]
    res = testee(A_4x4)

    assert ref.shape == res.shape
    assert np.all(ref == res)


@pytest.mark.skip(reason="Missing 'gather' translator.")
def test_slice_strides_vec():
    """Using strides.

    Note:
        Although we do not support the `strides` parameter of the `stride` primitive,
        this is not the reason why the test fails.
        It fails instead because Jax makes some strange gather stuff out of it.
    """

    A = np.arange(16)

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[1:15:2]

    ref = A[1:15:2]
    res = testee(A)

    assert ref.shape == res.shape
    assert np.all(ref == res)


@pytest.mark.skip(reason="Missing 'concatenate' translator.")
def test_slice_strides(A_4x4):
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


def test_slice_too_big(A_4x4):
    """Tests what happens if we specify a size that is too big.

    Note:
        It seems that the array is just returned as it is.
    """

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A[:20]

    res = testee(A_4x4)
    ref = A_4x4[:20]

    assert ref.shape == res.shape
    assert np.all(ref == res)


def test_dynamic_slice(A_4x4):
    def testee(A: np.ndarray, s1: int, s2: int) -> np.ndarray:
        return jax.lax.dynamic_slice(A, (s1, s2), (2, 2))

    ref = testee(A_4x4, 1, 1)

    with pytest.warns(
        expected_warning=UserWarning,
    ):
        res = jace.jit(testee)(A_4x4, 1, 1)

    assert np.all(ref == res)
