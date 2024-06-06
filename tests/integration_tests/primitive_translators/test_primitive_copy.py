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

from tests import util as testutil


def test_copy() -> None:
    @jace.jit
    def testee(A: np.ndarray) -> jax.Array:
        return jnp.copy(A)

    A = testutil.mkarray((10, 10, 10))
    res = testee(A)
    assert A.dtype == res.dtype
    assert A.shape == res.shape
    assert np.all(res == A)
