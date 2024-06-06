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


def test_iota_arange() -> None:
    """Tests `jnp.arange` functionality."""

    def testee(A: int) -> jax.Array:
        return jnp.arange(18, dtype=int) + A

    ref = testee(0)
    res = jace.jit(testee)(0)
    assert np.all(ref == res)


def test_iota_broadcast() -> None:
    """Test more iota using the `jax.lax.broadcasted_iota()` function."""
    shape = (2, 2, 2, 2)

    for d in range(len(shape)):
        # Must be inside the loop to bypass caching.
        def testee(A: np.int32) -> jax.Array:
            return jax.lax.broadcasted_iota("int32", shape, d) + A  # noqa: B023  # Variable capturing.

        ref = testee(np.int32(0))
        res = jace.jit(testee)(np.int32(0))

        assert res.shape == shape
        assert np.all(ref == res), f"Expected: {ref.tolist()}; Got: {res.tolist()}"
