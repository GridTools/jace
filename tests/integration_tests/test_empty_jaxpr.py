# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for empty jaxprs.

Todo:
    Add more tests that are related to `cond`, i.e. not all inputs are needed.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jace


def test_empty_single_return() -> None:
    @jace.jit
    def wrapped(A: np.ndarray) -> np.ndarray:
        return A

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    res = wrapped(A)

    assert np.all(res == A)
    assert res.__array_interface__["data"][0] != A.__array_interface__["data"][0]


def test_empty_multiple_return() -> None:
    @jace.jit
    def wrapped(A: np.ndarray, B: np.float64) -> tuple[np.ndarray, np.float64]:
        return A, B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.float64(30.0)
    res = wrapped(A, B)

    assert np.all(res[0] == A)
    assert res[1] == B
    assert res[0].__array_interface__["data"][0] != A.__array_interface__["data"][0]


def test_empty_unused_argument() -> None:
    """Empty body and an unused input argument."""

    @jace.jit
    def wrapped(A: np.ndarray, B: np.float64) -> np.ndarray:  # noqa: ARG001  # Explicitly unused.
        return A

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.float64(30.0)
    lowered = wrapped.lower(A, B)
    compiled = lowered.compile()
    res = compiled(A, B)

    assert len(lowered._translated_sdfg.inp_names) == 2
    assert len(compiled._csdfg.inp_names) == 2
    assert isinstance(res, np.ndarray)
    assert np.all(res == A)
    assert res.__array_interface__["data"][0] != A.__array_interface__["data"][0]


def test_empty_scalar() -> None:
    @jace.jit
    def wrapped(A: np.float64) -> np.float64:
        return A

    A = np.pi

    assert np.all(wrapped(A) == A)


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_empty_nested() -> None:
    @jace.jit
    def wrapped(A: np.float64) -> np.float64:
        return jax.jit(lambda A: A)(A)

    A = np.pi

    assert np.all(wrapped(A) == A)


@pytest.mark.skip(reason="Literal return value is not implemented.")
def test_empty_literal_return() -> None:
    """An empty Jaxpr that only contains a literal return value."""

    def testee() -> np.float64:
        return np.float64(3.1415)

    ref = testee()
    res = jace.jit(testee)()

    assert np.all(res == ref)


@pytest.mark.skip(reason="Literal return value is not implemented.")
def test_empty_with_drop_vars() -> None:
    """Jaxpr only containing drop variables.

    Notes:
        As a side effect the Jaxpr also has a literal return value.
    """

    @jace.grad
    def testee(a: np.float64, b: np.float64) -> np.float64:
        return a + b

    A = np.e
    ref = testee(A)
    res = jace.jit(testee)(A)

    assert np.all(ref == res)
