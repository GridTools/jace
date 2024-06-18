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
    def wrapped(a: np.ndarray) -> np.ndarray:
        return a

    a = np.arange(12, dtype=np.float64).reshape((4, 3))
    res = wrapped(a)

    assert np.all(res == a)
    assert res.__array_interface__["data"][0] != a.__array_interface__["data"][0]


def test_empty_multiple_return() -> None:
    @jace.jit
    def wrapped(a: np.ndarray, b: np.float64) -> tuple[np.ndarray, np.float64]:
        return a, b

    a = np.arange(12, dtype=np.float64).reshape((4, 3))
    b = np.float64(30.0)
    res = wrapped(a, b)

    assert np.all(res[0] == a)
    assert res[1] == b
    assert res[0].__array_interface__["data"][0] != a.__array_interface__["data"][0]


def test_empty_unused_argument() -> None:
    """Empty body and an unused input argument."""

    @jace.jit
    def wrapped(a: np.ndarray, b: np.float64) -> np.ndarray:  # noqa: ARG001  # Explicitly unused.
        return a

    a = np.arange(12, dtype=np.float64).reshape((4, 3))
    b = np.float64(30.0)
    lowered = wrapped.lower(a, b)
    compiled = lowered.compile()
    res = compiled(a, b)

    assert len(lowered._translated_sdfg.inp_names) == 2
    assert len(compiled._csdfg.inp_names) == 2
    assert isinstance(res, np.ndarray)
    assert np.all(res == a)
    assert res.__array_interface__["data"][0] != a.__array_interface__["data"][0]


def test_empty_scalar() -> None:
    @jace.jit
    def wrapped(a: np.float64) -> np.float64:
        return a

    a = np.float64(np.pi)

    assert np.all(wrapped(a) == a)


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_empty_nested() -> None:
    @jace.jit
    def wrapped(a: np.float64) -> np.float64:
        return jax.jit(lambda a: a)(a)

    a = np.float64(np.pi)

    assert np.all(wrapped(a) == a)


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

    a = np.e
    ref = testee(a)
    res = jace.jit(testee)(a)

    assert np.all(ref == res)
