# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from jax import numpy as jnp

import jace

from tests import util as testutil


def _perform_gather_test(
    testee: Callable,
    *args: Any,
) -> None:
    wrapped = jace.jit(testee)

    expected = testee(*args)
    result = wrapped(*args)

    assert np.allclose(expected, result)


def test_gather_simple_1():
    def testee(
        a: np.ndarray,
        idx: np.ndarray,
    ) -> np.ndarray:
        return a[idx]

    a = testutil.make_array(100)
    idx = testutil.make_array(300, dtype=np.int32, low=0, high=100)
    _perform_gather_test(testee, a, idx)


def test_gather_1():
    def testee(
        a: np.ndarray,
        idx: np.ndarray,
    ) -> np.ndarray:
        return a[idx, :, idx]

    a = testutil.make_array((300, 3, 300))
    idx = testutil.make_array(400, dtype=np.int32, low=1, high=300)
    _perform_gather_test(testee, a, idx)


def test_gather_2():
    def testee(
        a: np.ndarray,
        idx: np.ndarray,
    ) -> np.ndarray:
        return a[idx, :, :]

    a = testutil.make_array((300, 3, 300))
    idx = testutil.make_array(400, dtype=np.int32, low=1, high=300)
    _perform_gather_test(testee, a, idx)


def test_gather_3():
    def testee(
        a: np.ndarray,
        b: np.ndarray,
        idx: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        c = jnp.sin(a) + b
        return jnp.exp(c[idx, :, idx2])  # type: ignore[return-value]  # Type confusion.

    a = testutil.make_array((300, 3, 300))
    b = testutil.make_array((300, 3, 300))
    idx = testutil.make_array(400, dtype=np.int32, low=1, high=300)
    idx2 = testutil.make_array(400, dtype=np.int32, low=1, high=300)
    _perform_gather_test(testee, a, b, idx, idx2)
