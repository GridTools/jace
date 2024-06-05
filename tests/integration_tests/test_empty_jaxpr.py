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


def test_empty_array():
    @jace.jit
    def wrapped(A: np.ndarray) -> np.ndarray:
        return A

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    res = wrapped(A)

    assert np.all(res == A)
    assert res.__array_interface__["data"][0] != A.__array_interface__["data"][0]


def test_empty_multiple():
    @jace.jit
    def wrapped(A: np.ndarray, B: np.float64) -> tuple[np.ndarray, np.float64]:
        return A, B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.float64(30.0)
    res = wrapped(A, B)

    assert np.all(res[0] == A)
    assert res[1] == B
    assert res[0].__array_interface__["data"][0] != A.__array_interface__["data"][0]


def test_empty_unused():
    @jace.jit
    def wrapped(A: np.ndarray, B: np.float64) -> np.ndarray:  # noqa: ARG001  # Explicitly unused.
        return A

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.float64(30.0)
    lowered = wrapped.lower(A, B)
    compiled = lowered.compile()
    res = compiled(A, B)

    assert len(lowered._translated_sdfg.inp_names) == 2
    assert len(compiled._inp_names) == 2
    assert isinstance(res, np.ndarray)
    assert np.all(res == A)
    assert res.__array_interface__["data"][0] != A.__array_interface__["data"][0]


def test_empty_scalar():
    @jace.jit
    def wrapped(A: float) -> float:
        return A

    A = np.pi

    assert np.all(wrapped(A) == A)


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_empty_nested():
    @jace.jit
    def wrapped(A: float) -> float:
        return jax.jit(lambda A: A)(A)

    A = np.pi

    assert np.all(wrapped(A) == A)


def test_empty_with_drop_vars():
    """Tests if we can handle an empty input = output case, with present drop variables."""

    @jace.jit
    @jace.grad
    def wrapped(A: float) -> float:
        return A * A

    A = np.pi

    assert np.all(wrapped(A) == 2.0 * A)


@pytest.mark.skip(reason="Literal return value is not implemented.")
def test_empty_literal_return():
    """Tests if we can handle a literal return value.

    Note:
        Using this test function serves another purpose. Since Jax includes the original
        computation in the Jaxpr coming from a `grad` annotated function, the result will have
        only drop variables.

    Todo:
        Add a test if we really have a literal return value, but for that we need the Jaxpr.
    """

    @jace.jit
    @jace.grad
    def wrapped(A: float) -> float:
        return A + A + A

    A = np.e

    assert np.all(wrapped(A) == 3.0)