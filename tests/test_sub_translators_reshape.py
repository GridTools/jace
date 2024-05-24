# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the rehaping functionality."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from jax import numpy as jnp

import jace


def _test_impl_reshaping(
    src_shape: Sequence[int],
    dst_shape: Sequence[int],
    order: str = "C",
) -> None:
    """Performs a reshaping from `src_shape` to `dst_shape`."""
    A = np.random.random(src_shape)  # noqa: NPY002
    A = np.array(A, order=order)  # type: ignore[call-overload]  # MyPy wants a literal as order.

    def testee(A: np.ndarray) -> np.ndarray:
        return jnp.reshape(A, dst_shape)

    print(f"SHAPE: {A.shape} -> {dst_shape}")

    ref = testee(A)
    res = jace.jit(testee)(A)

    assert res.shape == dst_shape
    assert np.all(res == ref)


@pytest.fixture(
    params=["C", pytest.param("F", marks=pytest.mark.skip("Non C order is not supported"))]
)
def mem_order(request) -> str:
    """Gets the memory order that we want

    Currently 'F' is skipped because it is not implemented by the logic.
    """
    return request.param


def test_reshaping_same_rank(mem_order: str):
    """Keeping the ranke same."""
    _test_impl_reshaping((12, 2), (6, 4), mem_order)


def test_reshaping_adding_rank(mem_order: str):
    """Adding ranks to an array."""
    _test_impl_reshaping((12,), (12, 1), mem_order)
    _test_impl_reshaping((12,), (1, 12), mem_order)
    _test_impl_reshaping((12,), (1, 1, 12), mem_order)
    _test_impl_reshaping(
        (1,),
        (
            1,
            1,
        ),
        mem_order,
    )


def test_reshaping_removing_rank(mem_order: str):
    """Removing ranks from an array."""
    _test_impl_reshaping((12, 12), (144,), mem_order)
    _test_impl_reshaping(
        (
            1,
            1,
        ),
        (1,),
        mem_order,
    )
