# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the ALU translator.

Todo:
    - Add all supported primitives, to see if the template is valid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(
    params=[
        (jnp.logical_and, 2, np.bool_),
        (jnp.logical_or, 2, np.bool_),
        (jnp.logical_xor, 2, np.bool_),
        (jnp.logical_not, 1, np.bool_),
        (jnp.bitwise_and, 2, np.int64),
        (jnp.bitwise_or, 2, np.int64),
        (jnp.bitwise_xor, 2, np.int64),
        (jnp.bitwise_not, 1, np.int64),
    ]
)
def logical_ops(request) -> tuple[Callable, tuple[np.ndarray, ...]]:
    """Returns a logical operation function and inputs."""
    return (
        request.param[0],
        tuple(testutil.mkarray((2, 2), request.param[2]) for _ in range(request.param[1])),
    )


def _perform_alu_test(testee: Callable, *args: Any) -> None:
    """General function that just performs the test."""
    wrapped = jace.jit(testee)

    ref = testee(*args)
    res = wrapped(*args)
    assert np.allclose(ref, res), f"Expected '{ref.tolist()}' got '{res.tolist()}'"


def test_alu_unary_scalar():
    """Test unary ALU translator in the scalar case."""

    def testee(A: float) -> float | jax.Array:
        return jnp.cos(A)

    _perform_alu_test(testee, 1.0)


def test_alu_unary_array():
    """Test unary ALU translator with array argument."""

    def testee(A: np.ndarray) -> jax.Array:
        return jnp.sin(A)

    A = testutil.mkarray((100, 10, 3))

    _perform_alu_test(testee, A)


def test_alu_unary_scalar_literal():
    """Test unary ALU translator with literal argument"""

    def testee(A: float) -> float | jax.Array:
        return jnp.sin(1.98) + A

    _perform_alu_test(testee, 10.0)


def test_alu_unary_integer_power():
    """Tests the integer power, which has a parameter."""
    for exp in [0, 1, 2, 10]:

        def testee(A: np.ndarray) -> np.ndarray:
            return A ** int(exp)  # noqa: B023 # `exp` is not used in the body

        A = testutil.mkarray((10, 2 + exp, 3))
        _perform_alu_test(testee, A)


def test_alu_binary_scalar():
    """Scalar binary operation."""

    def testee(A: float, B: float) -> float:
        return A * B

    _perform_alu_test(testee, 1.0, 2.0)


def test_alu_binary_scalar_literal():
    """Scalar binary operation, with a literal."""

    def testee(A: float) -> float:
        return A * 2.03

    _perform_alu_test(testee, 7.0)


def test_alu_binary_scalar_literal_2():
    """Scalar binary operation, with a literal."""

    def testee(A: float) -> float:
        return 2.03 * A

    _perform_alu_test(testee, 7.0)


def test_alu_binary_array():
    """Test binary of arrays, with same size."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = testutil.mkarray((100, 10, 3))
    B = testutil.mkarray((100, 10, 3))
    _perform_alu_test(testee, A, B)


def test_alu_binary_array_scalar():
    """Test binary of array with scalar."""

    def testee(A: np.ndarray | float, B: float | np.ndarray) -> np.ndarray:
        return cast(np.ndarray, A + B)

    A = testutil.mkarray((100, 22))
    B = np.float64(1.34)
    _perform_alu_test(testee, A, B)
    _perform_alu_test(testee, B, A)


def test_alu_binary_array_literal():
    """Test binary of array with literal"""

    def testee(A: np.ndarray) -> np.ndarray:
        return A + 1.52

    A = testutil.mkarray((100, 22))
    _perform_alu_test(testee, A)


def test_alu_binary_array_literal_2():
    """Test binary of array with literal"""

    def testee(A: np.ndarray) -> np.ndarray:
        return 1.52 + A

    A = testutil.mkarray((100, 22))
    _perform_alu_test(testee, A)


def test_alu_binary_array_constants():
    """Test binary of array with constant."""

    def testee(A: np.ndarray) -> np.ndarray:
        return A + jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    A = testutil.mkarray((3, 3))
    _perform_alu_test(testee, A)


def test_alu_binary_broadcast_1():
    """Test broadcasting."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = testutil.mkarray((100, 1, 3))
    B = testutil.mkarray((100, 1, 1))
    _perform_alu_test(testee, A, B)
    _perform_alu_test(testee, B, A)


def test_alu_binary_broadcast_2():
    """Test broadcasting."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = testutil.mkarray((100, 1))
    B = testutil.mkarray((100, 10))
    _perform_alu_test(testee, A, B)
    _perform_alu_test(testee, B, A)


def test_alu_binary_broadcast_3():
    """Test broadcasting."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = testutil.mkarray((5, 1, 3, 4, 1))
    B = testutil.mkarray((5, 1, 3, 1, 2))
    _perform_alu_test(testee, A, B)
    _perform_alu_test(testee, B, A)


def test_alu_logical_bitwise_operation(
    logical_ops: tuple[Callable, tuple[np.ndarray, ...]],
):
    """Tests if the logical and bitwise operations works as they do in Jax."""
    inputs: tuple[np.ndarray, ...] = logical_ops[1]

    def testee(*args: np.ndarray) -> np.ndarray:
        return logical_ops[0](*args)

    _perform_alu_test(testee, *inputs)
