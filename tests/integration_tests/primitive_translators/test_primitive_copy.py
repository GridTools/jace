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
    def testee(a: np.ndarray) -> jax.Array:
        return jnp.copy(a)

    a = testutil.make_array((10, 10, 10))
    res = testee(a)
    assert a.dtype == res.dtype
    assert a.shape == res.shape
    assert a.__array_interface__["data"][0] != res.__array_interface__["data"][0]
    assert np.all(res == a)
