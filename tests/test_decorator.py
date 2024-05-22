# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the jit decorator.

Also see the `test_jax_api.py` test file, that tests composability.
"""

from __future__ import annotations

import numpy as np
import pytest

import jace


@pytest.fixture(autouse=True)
def _clear_translation_cache():
    """Decorator that clears the translation cache.

    Ensures that a function finds an empty cache and clears up afterwards.

    Todo:
        Should be used _everywhere_.
    """
    from jace.jax import translation_cache as tcache

    tcache.clear_translation_cache()
    yield
    tcache.clear_translation_cache()


def test_decorator_individually():
    """Tests the compilation steps individually."""

    def testee_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    lowering_cnt = [0]

    @jace.jit
    def testee(A, B):
        lowering_cnt[0] += 1
        return testee_(A, B)

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    lowered = testee.lower(A, B)
    compiled = lowered.compile()

    ref = testee_(A, B)
    res = compiled(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."
    assert lowering_cnt[0] == 1


def test_decorator_one_go():
    """Tests the compilation steps in one go."""

    def testee_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    lowering_cnt = [0]

    @jace.jit
    def testee(A, B):
        lowering_cnt[0] += 1
        return testee_(A, B)

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    ref = testee_(A, B)
    res = testee(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."
    assert lowering_cnt[0] == 1


def test_decorator_wrapped():
    """Tests if some properties are set correctly."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A * B

    wrapped = jace.jit(testee)

    assert wrapped.wrapped_fun is testee
    assert wrapped.__wrapped__ is testee
