# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace


def test_iota_arange() -> None:
    def testee(a: int) -> jax.Array:
        return jnp.arange(18, dtype=int) + a

    ref = testee(0)
    res = jace.jit(testee)(0)
    assert np.all(ref == res)


@pytest.mark.parametrize("d", [0, 1, 2, 3])
def test_iota_broadcast(d) -> None:
    shape = (2, 2, 2, 2)

    def testee(a: np.int32) -> jax.Array:
        return jax.lax.broadcasted_iota("int32", shape, d) + a

    ref = testee(np.int32(0))
    res = jace.jit(testee)(np.int32(0))

    assert res.shape == shape
    assert np.all(ref == res), f"Expected: {ref.tolist()}; Got: {res.tolist()}"
