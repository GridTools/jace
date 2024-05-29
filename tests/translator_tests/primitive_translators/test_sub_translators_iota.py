# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import jax
import numpy as np
from jax import numpy as jnp

import jace


def test_iota_arange():
    """Tests `jnp.arange` functionality."""

    def testee(A: int) -> np.ndarray:
        return jnp.arange(18, dtype=int) + A

    ref = testee(0)
    res = jace.jit(testee)(0)
    assert np.all(ref == res)


def test_iota_broadcast():
    """Test more iota using the `jax.lax.broadcasted_iota()` function."""
    shape = (4, 4, 4, 4)

    for d in range(len(shape)):

        def testee(A: np.int32) -> np.ndarray:
            return jax.lax.broadcasted_iota("int32", shape, d) + A  # noqa: B023  # Variable capturing.

        ref = testee(np.int32(0))
        res = jace.jit(testee)(np.int32(0))

        assert res.shape == shape
        assert np.all(ref == res)
