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
    def testee(A: np.ndarray) -> np.ndarray:
        return A

    A = np.arange(12, dtype=np.float64).reshape((4, 3))

    assert np.all(testee(A) == A)


def test_empty_scalar():
    @jace.jit
    def testee(A: float) -> float:
        return A

    A = np.pi

    assert np.all(testee(A) == A)


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_empty_nested():
    @jace.jit
    def testee3(A: float) -> float:
        return jax.jit(lambda A: A)(A)

    A = np.pi

    assert np.all(testee3(A) == A)
