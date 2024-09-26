# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the `select_n` translator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


@pytest.fixture(params=[True, False])
def pred(request) -> np.bool_:
    """Predicate used in the `test_mapped_unary_scalar_literal_*` tests."""
    return np.bool_(request.param)


def _perform_test(testee: Callable, *args: Any) -> None:
    res = testee(*args)
    ref = jace.jit(testee)(*args)
    assert np.all(res == ref)


def test_select_n_where() -> None:
    def testee(pred: np.ndarray, tbranch: np.ndarray, fbranch: np.ndarray) -> jax.Array:
        return jnp.where(pred, tbranch, fbranch)

    shape = (10, 10)
    pred = testutil.make_array(shape, np.bool_)
    tbranch = testutil.make_array(shape)
    fbranch = testutil.make_array(shape)
    _perform_test(testee, pred, tbranch, fbranch)


def test_select_n_where_literal_1(pred) -> None:
    def testee(pred: np.ndarray, fbranch: np.ndarray) -> jax.Array:
        return jnp.where(pred, 2, fbranch)

    fbranch = 1
    _perform_test(testee, pred, fbranch)


def test_select_n_where_literal_2(pred) -> None:
    def testee(pred: np.ndarray, tbranch: np.ndarray) -> jax.Array:
        return jnp.where(pred, tbranch, 3)

    tbranch = 2
    _perform_test(testee, pred, tbranch)


def test_select_n_where_literal_3(pred) -> None:
    def testee(pred: np.ndarray) -> jax.Array:
        return jnp.where(pred, 8, 9)

    _perform_test(testee, pred)


def test_select_n_many_inputs() -> None:
    """Tests the generalized way of using the primitive."""

    def testee(pred: np.ndarray, *cases: np.ndarray) -> jax.Array:
        return jax.lax.select_n(pred, *cases)

    nbcases = 10
    shape = (10, 10)
    cases = [np.full(shape, i) for i in range(nbcases)]
    pred = np.arange(cases[0].size).reshape(shape) % nbcases
    _perform_test(testee, pred, *cases)
