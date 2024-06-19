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


def test_cat_1d_arrays() -> None:
    """Concatenate two 1d arrays."""

    a1 = testutil.make_array(10)
    a2 = testutil.make_array(10)

    def testee(a1: np.ndarray, a2: np.ndarray) -> jax.Array:
        return jax.lax.concatenate((a1, a2), 0)

    ref = testee(a1, a2)
    res = jace.jit(testee)(a1, a2)

    assert res.shape == ref.shape
    assert np.all(ref == res)


def test_cat_nd() -> None:
    """Concatenate arrays of higher dimensions."""
    nb_arrays = 4
    std_shape: list[int] = [2, 3, 4, 5, 3]

    for cat_dim in range(len(std_shape)):
        tcache.clear_translation_cache()

        # Create the input that we ware using.
        input_arrays: list[np.ndarray] = []
        for _ in range(nb_arrays):
            shape = std_shape.copy()
            shape[cat_dim] = (testutil.make_array((), dtype=np.int32) % 10) + 1  # type: ignore[call-overload]  # type confusion
            input_arrays.append(testutil.make_array(shape))

        def testee(inputs: list[np.ndarray]) -> np.ndarray | jax.Array:
            return jax.lax.concatenate(inputs, cat_dim)  # noqa: B023  # Iteration variable capture.

        ref = testee(input_arrays)
        res = jace.jit(testee)(input_arrays)

        assert res.shape == ref.shape
        assert np.all(ref == res)


@pytest.mark.skip(reason="Jax does not support scalars as inputs.")
def test_cat_1d_array_scalars():
    """Concatenate an 1d array with scalars.

    This does not work, it is to observe Jax.
    """

    a1 = testutil.make_array(10)
    s1 = testutil.make_array(())
    s2 = testutil.make_array(())

    def testee(a1: np.ndarray, s1: np.float64, s2: np.float64) -> np.ndarray | jax.Array:
        return jnp.concatenate((s1, a1, s2), 0)

    ref = testee(a1, s1, s2)
    res = jace.jit(testee)(a1, s1, s2)

    assert res.shape == ref.shape
    assert np.all(ref == res)
