# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the ALU translator."""

from __future__ import annotations

import jax
import numpy as np

from jace import util as jutil


def test_add():
    """Simple add function."""
    jax.config.update("jax_enable_x64", True)

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    ref = testee(A, B)
    res = jutil._jace_run(testee, A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


if __name__ == "__main__":
    test_add()
