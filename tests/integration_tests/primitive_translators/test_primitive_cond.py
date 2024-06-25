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
from jace.util import translation_cache as tcache

from tests import util as testutil


def test_cond_simple1() -> None:

    def testee(val: np.float64, cond_arg: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return jax.lax.cond(
                0.5 > val.
                lambda arg: arg[0],
                lambda arg: jnp.array([13]) + arg[1],
                cond_arg,
        )

    vals: list[np.float64] = list(
