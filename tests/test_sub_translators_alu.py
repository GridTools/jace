# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the ALU translator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

import jace


def _perform_test(testee: Callable, *args: Any) -> None:
    """General function that just performs the test."""
    wrapped = jace.jit(testee)

    ref = testee(*args)
    res = wrapped(*args)
    assert np.allclose(ref, res), f"Expected '{ref.tolist()}' got '{res.tolist()}'"


def mkarr(
    shape: Sequence[int],
    dtype=np.float64,
) -> np.ndarray:
    return np.array(np.random.random(shape), dtype=dtype)  # noqa: NPY002


def test_alu_unary_scalar():
    """Test unary ALU translator in the scalar case."""

    def testee(A: float) -> float:
        return jnp.cos(A)

    _perform_test(testee, 1.0)


def test_alu_unary_array():
    """Test unary ALU translator with array argument."""

    def testee(A: np.ndarray) -> np.ndarray:
        return jnp.sin(A)

    A = mkarr((100, 10, 3))

    _perform_test(testee, A)


def test_alu_unary_scalar_literal():
    """Test unary ALU translator with literal argument"""

    def testee(A: float) -> float:
        return jnp.sin(1.98) + A

    _perform_test(testee, 10.0)


def test_alu_unary_integer_power():
    """Tests the integer power, which has a parameter."""
    for exp in [0, 1, 2, 10]:

        def testee(A: np.ndarray) -> np.ndarray:
            return A ** int(exp)  # noqa: B023 # `exp` is not used in the body

        A = mkarr((10, 2 + exp, 3))
        _perform_test(testee, A)


def test_alu_binary_scalar():
    """Scalar binary operation."""

    def testee(A: float, B: float) -> float:
        return A * B

    _perform_test(testee, 1.0, 2.0)


def test_alu_binary_scalar_literal():
    """Scalar binary operation, with a literal."""

    def testee(A: float) -> float:
        return A * 2.03

    _perform_test(testee, 7.0)


def test_alu_binary_array():
    """Test binary of arrays, with same size."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = mkarr((100, 10, 3))
    B = mkarr((100, 10, 3))
    _perform_test(testee, A, B)


def test_alu_binary_array_scalar():
    """Test binary of array with scalar."""

    def testee(A: np.ndarray, B: float) -> np.ndarray:
        return A + B

    A = mkarr((100, 22))
    B = np.float64(1.34)
    _perform_test(testee, A, B)


def test_alu_binary_array_literal():
    """Test binary of array with literal"""

    def testee(A: np.ndarray) -> np.ndarray:
        return A + 1.52

    A = mkarr((100, 22))
    _perform_test(testee, A)


def test_alu_binary_array_literal_2():
    """Test binary of array with literal"""

    def testee(A: np.ndarray) -> np.ndarray:
        return 1.52 + A

    A = mkarr((100, 22))
    _perform_test(testee, A)


def test_alu_binary_array_constants():
    """Test binary of array with constant."""

    def testee(A: np.ndarray) -> np.ndarray:
        return A + jax.numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    A = mkarr((3, 3))
    _perform_test(testee, A)


def test_alu_binary_broadcast_1():
    """Test broadcasting."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = mkarr((100, 1, 3))
    B = mkarr((100, 1, 1))
    _perform_test(testee, A, B)
    _perform_test(testee, B, A)


def test_alu_binary_broadcast_2():
    """Test broadcasting."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = mkarr((100, 1))
    B = mkarr((100, 10))
    _perform_test(testee, A, B)
    _perform_test(testee, B, A)
