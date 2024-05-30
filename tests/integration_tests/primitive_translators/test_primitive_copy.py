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


def test_copy():
    @jace.jit
    def testee(A: np.ndarray) -> jax.Array:
        return jnp.copy(A)

    A = np.random.random((10, 10, 10))  # noqa: NPY002
    ref = np.copy(A)
    res = testee(A)
    assert ref.dtype == res.dtype
    assert np.all(ref == res)
