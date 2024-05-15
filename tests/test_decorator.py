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


def test_decorator_annotation():
    """Tests the annotation, essential `jace.jax.api_helper.jax_wrapper`."""
    assert jax.jit.__doc__ == jace.jit.__doc__


def test_decorator_individually():
    """Tests the compilation steps individually."""
    jax.config.update("jax_enable_x64", True)

    def testee_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    @jace.jit
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return testee_(A, B)

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    lowered = testee.lower(A, B)
    compiled = lowered.compile()

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


def test_decorator_caching():
    """This tests the caching ability"""
    jax.config.update("jax_enable_x64", True)

    def testee1_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A * B

    def testee2_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    testee1 = jace.jit(testee1_)
    testee2 = jace.jit(testee2_)

    assert testee1.__wrapped__ == testee1_
    assert testee2.__wrapped__ == testee2_

    # This is the first size
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # This is the second sizes
    C = np.arange(16, dtype=np.float64).reshape((4, 4))
    D = np.full((4, 4), 10, dtype=np.float64)

    # Lower the two functions for the first size.
    lowered1_size1 = testee1.lower(A, B)
    lowered2_size1 = testee2.lower(A, B)

    # If we now lower them again, we should get the same objects
    assert lowered1_size1 is testee1.lower(A, B)
    assert lowered2_size1 is testee2.lower(A, B)

    # Now we lower them for the second sizes.
    lowered1_size2 = testee1.lower(C, D)
    lowered2_size2 = testee2.lower(C, D)

    # Again if we now lower them again, we should get the same objects.
    assert lowered1_size1 is testee1.lower(A, B)
    assert lowered2_size1 is testee2.lower(A, B)
    assert lowered1_size2 is testee1.lower(C, D)
    assert lowered2_size2 is testee2.lower(C, D)

    # Now use the compilation; since all is the same code path we only use one size.
    compiled1 = lowered1_size1.compile()
    compiled2 = lowered1_size1.compile({"dummy_option": True})

    assert compiled1 is lowered1_size1.compile()
    assert compiled2 is lowered1_size1.compile({"dummy_option": True})
    assert compiled2 is not lowered1_size1.compile({"dummy_option": False})
    assert compiled2 is lowered1_size1.compile({"dummy_option": True})


def test_decorator_sharing():
    """Tests if there is no false sharing in the cache."""
    jax.config.update("jax_enable_x64", True)

    @jace.jit
    def jaceWrapped(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        C = A * B
        D = C + A
        E = D + B  # Just enough state.
        return A + B + C + D + E

    # These are the argument
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # Now we lower it.
    jaceLowered = jaceWrapped.lower(A, B)

    # Now we compile it with enabled optimization.
    optiCompiled = jaceLowered.compile({"auto_optimize": True, "simplify": True})

    # Now we compile it without any optimization.
    unoptiCompiled = jaceLowered.compile({})

    # Because of the way how things work the optimized must have more than the unoptimized.
    #  If there is sharing, then this would not be the case.
    assert optiCompiled._csdfg.sdfg.number_of_nodes() == 1
    assert optiCompiled._csdfg.sdfg.number_of_nodes() < unoptiCompiled._csdfg.sdfg.number_of_nodes()
