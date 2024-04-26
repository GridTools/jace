# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the compability of the JaCe api to Jax."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace
from jace import util as jutil


np.random.seed(42)


def test_jit():
    """Simple add function."""
    jax.config.update("jax_enable_x64", True)

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    jax_testee = jax.jit(testee)
    jace_testee = jace.jit(testee)

    assert jutil.is_jaxified(jax_testee)
    assert not jutil.is_jaxified(jace_testee)
    assert not jutil.is_jaceified(jax_testee)
    assert jutil.is_jaceified(jace_testee)

    ref = jax_testee(A, B)
    res = jace_testee(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


@pytest.mark.skip(reason="Scalar return values are not handled.")
def test_composition1():
    jax.config.update("jax_enable_x64", True)

    def f_(x):
        return jnp.sin(x)

    def df_(x):
        return jnp.cos(x)

    def ddf_(x):
        return -jnp.sin(x)

    x = 1.0

    # Jacify it.
    f = jace.jit(f_)
    assert jutil.is_jaceified(f)
    assert not jutil.is_jaxified(f)

    ref = f_(x)
    res = f(x)
    assert np.allclose(ref, res), f"f: Expected '{ref}', got '{res}'."

    # Now apply a Jax transformation to the jaceified function.
    df = jax.grad(f)

    ref = df_(x)
    res = df(x)
    assert np.allclose(ref, res), f"df: Expected '{ref}', got '{res}'."

    # Now apply a jace transformation around a jaxified transformation.
    ddf = jace.grad(df)

    ref = ddf_(x)
    res = ddf(x)
    assert np.allclose(ref, res), f"ddf: Expected '{ref}', got '{res}'."


def test_composition2():
    jax.config.update("jax_enable_x64", True)

    def f1_(A, B):
        return A + B

    f1 = jax.jit(f1_)

    def f2_(A, B, C):
        return f1(A, B) - C

    f2 = jace.jit(f2_)

    def f3_(A, B, C, D):
        return f2(A, B, C) * D

    f3_jax = jax.jit(f3_)
    f3_jace = jace.jit(f3_)

    A, B, C, D = (np.random.random((10, 3, 50)) for _ in range(4))

    ref = ((A + B) - C) * D

    # We have to disable it, because currently there is no `pjit` instruction
    #  that can handle the nesting.
    with jax.disable_jit():
        res_jax = f3_jax(A, B, C, D)
        res_jace = f3_jace(A, B, C, D)

    assert np.allclose(ref, res_jax), "Jax failed."
    assert np.allclose(ref, res_jace), "JaCe Failed."


if __name__ == "__main__":
    test_jit()
    # test_composition1()
    test_composition2()
