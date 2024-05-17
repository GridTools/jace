# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for empty jaxprs.
."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jace


def test_empty_array():
    jax.config.update("jax_enable_x64", True)

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A

    A = np.arange(12, dtype=np.float64).reshape((4, 3))

    assert np.all(testee(A) == A)


@pytest.mark.skip(reason="Scalar return values are not handled.")
def test_empty_scalar():
    jax.config.update("jax_enable_x64", True)

    @jace.jit
    def testee(A: float) -> float:
        return A

    A = np.pi

    assert np.all(testee(A) == A)


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_empty_nested():
    jax.config.update("jax_enable_x64", True)

    @jace.jit
    def testee3(A: float) -> float:
        return jax.jit(lambda A: A)(A)

    A = np.pi

    assert np.all(testee3(A) == A)
