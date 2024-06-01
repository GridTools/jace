# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the `select_n` translator."""

from __future__ import annotations

from typing import Any, Callable

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


@pytest.fixture()
def Pred() -> np.ndarray:
    return testutil.mkarray((10, 10)) > 0.5


@pytest.fixture()
def tbranch() -> np.ndarray:
    return np.ones((10, 10))


@pytest.fixture()
def fbranch() -> np.ndarray:
    return np.zeros((10, 10))


def _perform_test(testee: Callable, *args: Any):

    res = testee(*args)
    ref = jace.jit(testee)(*args)

    assert args[0].shape == res.shape
    assert np.all(res == ref)


def test_select_n_where(Pred, tbranch, fbranch):
    """Normal `np.where` test."""

    def testee(P: Any, T: Any, F: Any) -> Any:
        return jnp.where(P, T, F)
    _perform_test(testee, Pred, tbranch, fbranch)


def test_select_n_where_one_literal(Pred, tbranch, fbranch):
    """`np.where` where one of the input is a literal.
    """

    def testee1(P: Any, F: Any) -> Any:
        return jnp.where(P, 2, F)

    def testee2(P: Any, T: Any) -> Any:
        return jnp.where(P, T, 3)

    _perform_test(testee1, Pred, fbranch)
    _perform_test(testee2, Pred, tbranch)


def test_select_n_where_full_literal(Pred):
    """`np.where` where all inputs are literals."""

    def testee(P: Any) -> Any:
        return jnp.where(P, 8, 9)

    # If not a scalar, Jax will do broadcasting and no literal substitution is done.
    Pred = Pred[0, 0]
    _perform_test(testee, Pred)


def test_select_n_many_inputs():
    """Tests the generalized way of using the primitive."""
    nbcases = 10
    shape = (10, 10)
    cases = [np.full(shape, i) for i in range(nbcases)]
    pred = np.arange(cases[0].size).reshape(shape) % nbcases

    def testee(pred: np.ndarray, *cases: np.ndarray) -> jax.Array:
        return jax.lax.select_n(pred, *cases)

    ref = testee(pred, *cases)
    res = jace.jit(testee)(pred, *cases)

    assert np.all(ref == res)
