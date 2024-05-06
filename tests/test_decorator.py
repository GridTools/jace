# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the jit decorator.

Also see the `test_jax_api.py` test file, that tests composability.
."""

from __future__ import annotations

import jax
import numpy as np

import jace


def test_decorator_individually():
    """Tests the compilation steps individually."""
    jax.config.update("jax_enable_x64", True)

    def testee_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    testee = jace.jit(testee_)
    lowered = testee.lower(A, B)
    optimized = lowered.optimize()
    compiled = optimized.compile()

    ref = testee_(A, B)
    res = compiled(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


def test_decorator_one_go():
    """Tests the compilation steps in one go."""
    jax.config.update("jax_enable_x64", True)

    def testee_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    testee = jace.jit(testee_)

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    ref = testee_(A, B)
    res = testee(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."
