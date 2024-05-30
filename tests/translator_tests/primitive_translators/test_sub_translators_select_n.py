# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the `select_n` translator."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace


@pytest.fixture(autouse=True)
def _disable_jit():
    """Decorator that ensures that `select_n` is not put in an implicit `jit`.

    The reason we do this is because we can currently not handle this nested jits.
    It is important that it also disabled explicit usage of `jax.jit`.
    However, since JaCe does not honor this flag we it does not affect us.

    Todo:
        Remove as soon as we can handle nested `jit`.
    """
    with jax.disable_jit(disable=True):
        yield


@pytest.fixture(autouse=True)
def _enable_x64_mode_in_jax():
    """Ensures that x64 mode in Jax ins enabled."""
    with jax.experimental.enable_x64():
        yield


@pytest.fixture()
def Pred() -> np.ndarray:
    return np.random.random((10, 10)) > 0.5  # noqa: NPY002


@pytest.fixture()
def tbranch() -> np.ndarray:
    return np.ones((10, 10))


@pytest.fixture()
def fbranch() -> np.ndarray:
    return np.zeros((10, 10))


def _perform_test(P: Any, T: Any, F: Any):
    def testee(P: Any, T: Any, F: Any):
        return jnp.where(P, T, F)

    res = testee(P, T, F)
    ref = jace.jit(testee)(P, T, F)

    assert np.all(res == ref)


def test_select_n_where(Pred, tbranch, fbranch):
    """Normal `np.where` test."""
    _perform_test(Pred, tbranch, fbranch)


def test_select_n_where_one_literal(Pred, tbranch, fbranch):
    """`np.where` where one of the input is a literal."""
    _perform_test(Pred, 2, fbranch)
    _perform_test(Pred, tbranch, 3)


def test_select_n_where_full_literal(Pred):
    """`np.where` where all inputs are literals."""
    _perform_test(Pred, 8, 9)


def test_select_n_many_inputs():
    """Tests the generalized way of using the primitive."""
    nbcases = 5
    shape = (10, 10)
    cases = [np.full(shape, i) for i in range(nbcases)]
    pred = np.arange(cases[0].size).reshape(shape) % 5

    def testee(pred: np.ndarray, *cases: np.ndarray) -> jax.Array:
        return jax.lax.select_n(pred, *cases)

    ref = testee(pred, *cases)
    res = jace.jit(testee)(pred, *cases)

    assert np.all(ref == res)
