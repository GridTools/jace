# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the broadcast in dim translator.

Parts of the tests are also implemented inside `test_sub_translators_squeeze_expand_dims.py`.

Todo:
    - `np.meshgrid`
    - `np.ix_`
    - `np.indices`
"""

from __future__ import annotations

import numpy as np
from jax import numpy as jnp

import jace


def test_bid_scalar():
    """Broadcast a scalar to a matrix."""

    def testee(A: float) -> np.ndarray:
        return jnp.broadcast_to(A, (2, 2))

    for a in [1, 1.0, 3.1415]:
        ref = testee(a)
        res = jace.jit(testee)(a)

        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        assert np.all(res == ref), f"Expected '{ref.tolist()}' got '{res.tolist()}'."


def test_bid_literal():
    """Broadcast a literal to a matrix."""

    def testee(a: float) -> np.ndarray:
        return jnp.broadcast_to(1.0, (10, 10)) + a

    for a in [1, 1.0, 3.1415]:
        ref = testee(a)
        res = jace.jit(testee)(a)
        assert res.shape == ref.shape
        assert res.dtype == ref.dtype
        assert np.all(res == ref)
