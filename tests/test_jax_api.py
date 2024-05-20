# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the compatibility of the JaCe api to Jax."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace
from jace import util as jutil


np.random.seed(42)  # noqa: NPY002  # random generator


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
def test_composition_itself():
    """Tests if Jace is composable with itself."""
    jax.config.update("jax_enable_x64", True)

    # Pure Python functions
    def f_ref(x):
        return jnp.sin(x)

    def df_ref(x):
        return jnp.cos(x)

    def ddf_ref(x):
        return -jnp.sin(x)

    # Annotated functions.

    @jace.jit
    def f(x):
        return f_ref(x)

    @jace.jit
    def df(x):
        return jace.grad(f)(x)

    @jace.jit
    @jace.grad
    def ddf(x):
        return df(x)

    assert all(jutil.is_jaceified(x) for x in [f, df, ddf])

    x = 1.0
    for fun, fref in zip([f, df, ddf], [f_ref, df_ref, ddf_ref]):
        ref = fref(x)
        res = fun(x)
        assert np.allclose(ref, res), f"f: Expected '{ref}', got '{res}'."


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_composition_with_jax():
    """Tests if Jace can interact with Jax and vice versa."""
    jax.config.update("jax_enable_x64", True)

    def base_fun(A, B, C):
        return A + B * jnp.sin(C) - A * B

    @jace.jit
    def jace_fun(A, B, C):
        return jax.jit(base_fun)(A, B, C)

    def jax_fun(A, B, C):
        return jace.jit(base_fun)(A, B, C)

    A, B, C = (np.random.random((10, 3, 50)) for _ in range(3))  # noqa: NPY002  # random generator

    assert np.allclose(jace_fun(A, B, C), jax_fun(A, B, C))


@pytest.mark.skip(reason="Nested Jaxpr are not handled.")
def test_composition_with_jax_2():
    """Second test if Jace can interact with Jax and vice versa."""

    @jax.jit
    def f1_jax(A, B):
        return A + B

    assert jutil.is_jaxified(f1_jax)

    @jace.jit
    def f2_jace(A, B, C):
        return f1_jax(A, B) - C

    assert jutil.is_jaceified(f2_jace)

    @jax.jit
    def f3_jax(A, B, C, D):
        return f2_jace(A, B, C) * D

    assert jutil.is_jaxified(f3_jax)

    @jace.jit
    def f3_jace(A, B, C, D):
        return f3_jax(A, B, C, D)

    assert jutil.is_jaceified(f3_jace)

    A, B, C, D = (np.random.random((10, 3, 50)) for _ in range(4))  # noqa: NPY002  # random generator

    ref = ((A + B) - C) * D

    res_jax = f3_jax(A, B, C, D)
    res_jace = f3_jace(A, B, C, D)

    assert np.allclose(ref, res_jax), "Jax failed."
    assert np.allclose(ref, res_jace), "JaCe Failed."


@pytest.mark.skip(reason="Scalar return values are not handled.")
def test_grad_annotation_direct():
    """Test if `jace.grad` works directly."""
    jax.config.update("jax_enable_x64", True)

    def f(x):
        return jnp.sin(jnp.exp(jnp.cos(x**2)))

    @jax.grad
    def jax_df(x):
        return f(x)

    @jax.jit
    def jace_df(x):
        return jace.grad(f)(x)

    # These are the random numbers where we test
    Xs = (np.random.random(10) - 0.5) * 10  # noqa: NPY002  # Random number generator

    for i in range(Xs.shape[0]):
        x = Xs[i]
        res = jace_df(x)
        ref = jax_df(x)
        assert np.allclose(res, ref)


def test_grad_control_flow():
    """Tests if `grad` and controlflow works.

    This requirement is mentioned in `https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-autodiff`.
    """
    jax.config.update("jax_enable_x64", True)

    @jace.grad
    def df(x):
        if x < 3:
            return 3.0 * x**2
        return -4 * x

    x1 = 2.0
    df_x1 = 6 * x1
    x2 = 4.0
    df_x2 = -4.0

    res_1 = df(x1)
    res_2 = df(x2)

    assert df(x1) == df_x1, f"Failed lower branch, expected '{df_x1}', got '{res_1}'."
    assert df(x2) == df_x2, f"Failed upper branch, expected '{df_x2}', got '{res_2}'."
