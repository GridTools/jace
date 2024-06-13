# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the `broadcast_in_dim` primitive.

Parts of the tests are also implemented inside
`test_sub_translators_squeeze_expand_dims.py`, because this primitive has a relation to
`squeeze`.

Todo:
    - `np.meshgrid`
    - `np.ix_`
    - `np.indices`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture(params=[(10,), (10, 1), (1, 10)])
def vector_shape(request) -> tuple[int, ...]:
    """Shapes used in the `test_bid_vector()` tests."""
    return request.param


def test_bid_scalar() -> None:
    """Broadcast a scalar to a matrix."""

    def testee(A: float) -> jax.Array:
        return jnp.broadcast_to(A, (2, 2))

    A = 1.032
    ref = testee(A)
    res = jace.jit(testee)(A)

    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert np.all(res == ref), f"Expected '{ref.tolist()}' got '{res.tolist()}'."


def test_bid_literal() -> None:
    """Broadcast a literal to a matrix."""

    def testee(a: float) -> jax.Array:
        return jnp.broadcast_to(1.0, (10, 10)) + a

    ref = testee(0.0)
    res = jace.jit(testee)(0.0)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert np.all(res == ref)


def test_bid_vector(vector_shape: Sequence[int]) -> None:
    """Broadcast a vector to a tensor."""

    def testee(A: np.ndarray) -> jax.Array:
        return jnp.broadcast_to(A, (10, 10))

    A = testutil.mkarray(vector_shape)
    ref = testee(A)
    res = jace.jit(testee)(A)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert np.all(res == ref)
