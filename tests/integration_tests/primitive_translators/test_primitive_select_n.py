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


def _perform_test(testee: Callable, *args: Any):

    res = testee(*args)
    ref = jace.jit(testee)(*args)
    assert np.all(res == ref)


def test_select_n_where():
    """Normal `np.where` test."""

    def testee(P: Any, T: Any, F: Any) -> Any:
        return jnp.where(P, T, F)

    shape = (10, 10)
    pred = testutil.mkarray(shape, np.bool_)
    tbranch = testutil.mkarray(shape)
    fbranch = testutil.mkarray(shape)
    _perform_test(testee, pred, tbranch, fbranch)


def test_select_n_where_literal():
    """`np.where` where one of the input is a literal.
    """

    def testee1(P: Any, F: Any) -> Any:
        return jnp.where(P, 2, F)

    def testee2(P: Any, T: Any) -> Any:
        return jnp.where(P, T, 3)

    def testee3(P: Any) -> Any:
        return jnp.where(P, 8, 9)

    shape = ()
    pred = testutil.mkarray(shape, np.bool_)
    tbranch = testutil.mkarray(shape, np.int_)
    fbranch = testutil.mkarray(shape, np.int_)

    _perform_test(testee1, pred, fbranch)
    _perform_test(testee2, pred, tbranch)
    _perform_test(testee3, pred)


def test_select_n_many_inputs():
    """Tests the generalized way of using the primitive."""

    def testee(pred: np.ndarray, *cases: np.ndarray) -> jax.Array:
        return jax.lax.select_n(pred, *cases)

    nbcases = 10
    shape = (10, 10)
    cases = [np.full(shape, i) for i in range(nbcases)]
    pred = np.arange(cases[0].size).reshape(shape) % nbcases
    _perform_test(testee, pred, *cases)
