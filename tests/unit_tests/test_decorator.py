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

import jace

from tests import util as testutil


def test_decorator_individually() -> None:
    """Tests the compilation steps individually."""

    def testee_(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    lowering_cnt = [0]

    @jace.jit
    def testee(a, b):
        lowering_cnt[0] += 1
        return testee_(a, b)

    a = testutil.make_array((4, 3))
    b = testutil.make_array((4, 3))

    lowered = testee.lower(a, b)
    compiled = lowered.compile()

    ref = testee_(a, b)
    res = compiled(a, b)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."
    assert lowering_cnt[0] == 1


def test_decorator_one_go() -> None:
    """Tests the compilation steps in one go."""

    def testee_(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    lowering_cnt = [0]

    @jace.jit
    def testee(a, b):
        lowering_cnt[0] += 1
        return testee_(a, b)

    a = testutil.make_array((4, 3))
    b = testutil.make_array((4, 3))

    ref = testee_(a, b)
    res = testee(a, b)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."
    assert lowering_cnt[0] == 1


def test_decorator_wrapped() -> None:
    """Tests if some properties are set correctly."""

    def testee(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    wrapped = jace.jit(testee)

    assert wrapped.wrapped_fun is testee
