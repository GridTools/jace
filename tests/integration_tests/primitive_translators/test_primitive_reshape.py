# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the rehaping functionality."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


def _test_impl_reshaping(
    src_shape: Sequence[int], dst_shape: Sequence[int], order: str = "C"
) -> None:
    """Performs a reshaping from `src_shape` to `dst_shape`."""
    a = testutil.make_array(src_shape)
    a = np.array(a, order=order)  # type: ignore[call-overload]  # MyPy wants a literal as order.

    def testee(a: np.ndarray) -> jax.Array:
        return jnp.reshape(a, dst_shape)

    ref = testee(a)
    res = jace.jit(testee)(a)

    assert res.shape == dst_shape
    assert np.all(res == ref)


@pytest.fixture(params=["C", "F"])
def mem_order(request) -> str:
    """Gets the memory order that we want."""
    return request.param


@pytest.fixture(params=[(216, 1, 1), (1, 216, 1), (1, 1, 216), (1, 6, 36), (36, 1, 6)])
def new_shape(request) -> None:
    """New shapes for the `test_reshaping_same_rank()` test."""
    return request.param


@pytest.fixture(params=[(12, 1), (1, 12), (1, 1, 12), (1, 2, 6)])
def expanded_shape(request) -> None:
    """New shapes for the `test_reshaping_removing_rank()` test."""
    return request.param


@pytest.fixture(params=[(216,), (6, 36), (36, 6), (216, 1)])
def reduced_shape(request) -> None:
    """New shapes for the `test_reshaping_adding_rank()` test."""
    return request.param


def test_reshaping_same_rank(new_shape: Sequence[int], mem_order: str) -> None:
    """The rank, numbers of dimensions, stays the same,"""
    _test_impl_reshaping((6, 6, 6), new_shape, mem_order)


def test_reshaping_adding_rank(expanded_shape: Sequence[int], mem_order: str) -> None:
    """Adding ranks to an array."""
    _test_impl_reshaping((12,), expanded_shape, mem_order)


def test_reshaping_removing_rank(reduced_shape: Sequence[int], mem_order: str) -> None:
    """Removing ranks from an array."""
    _test_impl_reshaping((6, 6, 6), reduced_shape, mem_order)
