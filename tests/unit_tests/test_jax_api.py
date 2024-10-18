# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the compatibility of the JaCe api to JAX."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import numpy as jnp, tree_util as jax_tree

import jace
from jace import translated_jaxpr_sdfg as tjsdfg, translator, util
from jace.translator import post_translation as ptranslation

from tests import util as testutil


def test_jit() -> None:
    """Simple add function."""

    def testee(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    a = testutil.make_array((4, 3))
    b = testutil.make_array((4, 3))

    jax_testee = jax.jit(testee)
    jace_testee = jace.jit(testee)

    ref = jax_testee(a, b)
    res = jace_testee(a, b)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


def test_composition_itself() -> None:
    """Tests if JaCe is composable with itself."""

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

    assert all(isinstance(x, jace.stages.JaCeWrapped) for x in [f, df, ddf])

    x = 1.0
    for fun, fref in zip([f, df, ddf], [f_ref, df_ref, ddf_ref]):
        ref = fref(x)
        res = fun(x)
        assert np.allclose(ref, res), f"f: Expected '{ref}', got '{res}'."


def test_composition_with_jax() -> None:
    """Tests if JaCe can interact with JAX and vice versa."""

    def base_fun(a, b, c):
        return a + b * jnp.sin(c) - a * b

    @jace.jit
    def jace_fun(a, b, c):
        return jax.jit(base_fun)(a, b, c)

    def jax_fun(a, b, c):
        return jace.jit(base_fun)(a, b, c)

    a, b, c = (testutil.make_array((10, 3, 50)) for _ in range(3))

    assert np.allclose(jace_fun(a, b, c), jax_fun(a, b, c))


def test_composition_with_jax_2() -> None:
    """Second test if JaCe can interact with JAX and vice versa."""

    @jax.jit
    def f1_jax(a, b):
        return a + b

    @jace.jit
    def f2_jace(a, b, c):
        return f1_jax(a, b) - c

    @jax.jit
    def f3_jax(a, b, c, d):
        return f2_jace(a, b, c) * d

    @jace.jit
    def f3_jace(a, b, c, d):
        return f3_jax(a, b, c, d)

    a, b, c, d = (testutil.make_array((10, 3, 50)) for _ in range(4))

    ref = ((a + b) - c) * d
    res_jax = f3_jax(a, b, c, d)
    res_jace = f3_jace(a, b, c, d)

    assert np.allclose(ref, res_jax), "JAX failed."
    assert np.allclose(ref, res_jace), "JaCe Failed."


def test_grad_annotation_direct() -> None:
    """Test if `jace.grad` works directly."""

    def f(x):
        return jnp.sin(jnp.exp(jnp.cos(x**2)))

    @jax.grad
    def jax_ddf(x):
        return jax.grad(f)(x)

    @jax.jit
    def jace_ddf(x):
        return jace.grad(jace.grad(f))(x)

    # These are the random numbers where we test
    xs = (testutil.make_array(10) - 0.5) * 10

    for i in range(xs.shape[0]):
        x = xs[i]
        res = jace_ddf(x)
        ref = jax_ddf(x)
        assert np.allclose(res, ref)


def test_grad_control_flow() -> None:
    """Tests if `grad` and controlflow works.

    This requirement is mentioned in the [documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-autodiff).
    """

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


def test_disabled_x64() -> None:
    """Tests the behaviour of the tool chain if x64 support is disabled.

    Notes:
        Once the x64 issue is resolved make this test a bit more useful.
    """

    def testee(a: np.ndarray, b: np.float64) -> np.ndarray:
        return a + b

    a = testutil.make_array((4, 3))
    b = np.float64(10.0)

    # Run them with disabled x64 support
    #  This is basically a reimplementation of the `JaCeWrapped.lower()` function.
    #  but we have to do it this way to disable the x64 mode in translation.
    with jax.experimental.disable_x64():
        jaxpr = jax.make_jaxpr(testee)(a, b)

    flat_call_args = jax_tree.tree_leaves(((a, b), {}))
    builder = translator.JaxprTranslationBuilder(
        primitive_translators=translator.get_registered_primitive_translators()
    )
    trans_ctx: translator.TranslationContext = builder.translate_jaxpr(jaxpr)

    tsdfg: tjsdfg.TranslatedJaxprSDFG = ptranslation.postprocess_jaxpr_sdfg(
        trans_ctx=trans_ctx, fun=testee, flat_call_args=flat_call_args
    )

    # Because x64 is disabled JAX traces the input as float32, even if we have passed
    #  float64 as input! Calling the resulting SDFG with the arguments we used for
    #  lowering will result in an error, because of the situation,
    #  `sizeof(float32) < sizeof(float64)`, no out of bound error would result, but the
    #  values are garbage.
    assert all(
        tsdfg.sdfg.arrays[input_name].dtype.as_numpy_dtype().type is np.float32
        for input_name in tsdfg.input_names
    )


@pytest.mark.usefixtures("_enable_jit")
def test_tracing_detection() -> None:
    """Tests our ability to detect if tracing is going on."""
    expected_tracing_state = False

    def testee(a: float, b: int) -> float:
        c = a + b
        assert util.is_tracing_ongoing(a, b) == expected_tracing_state
        assert util.is_tracing_ongoing() == expected_tracing_state
        return a + c

    # We do not expect tracing to happen.
    _ = testee(1.0, 1)

    # Now tracing is going on
    expected_tracing_state = True
    _ = jax.jit(testee)(1.0, 1)
    _ = jace.jit(testee)(1.0, 1)

    # Tracing should now again be disabled
    expected_tracing_state = False
    _ = testee


def test_no_input() -> None:
    """Tests if we can handle the case of no input."""

    @jace.jit
    def ones10x10() -> jax.Array:
        return jnp.ones((10, 10), dtype=np.int32)

    res = ones10x10()

    assert res.shape == (10, 10)
    assert res.dtype == np.int32
    assert np.all(res == 1)


def test_jax_array_as_input() -> None:
    """This function tests if we use JAX arrays as inputs."""

    def testee(a: jax.Array) -> jax.Array:
        return jnp.sin(a + 1.0)

    a = jnp.array(testutil.make_array((10, 19)))

    ref = testee(a)
    res = jace.jit(testee)(a)

    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert np.allclose(res, ref)


def test_jax_pytree() -> None:
    """Perform if pytrees are handled correctly."""

    def testee(a: dict[str, np.ndarray]) -> dict[str, jax.Array]:
        mod_a = {k: jnp.sin(v) for k, v in a.items()}
        mod_a["__additional"] = jnp.asin(a["a1"])
        return mod_a

    a = {f"a{i}": testutil.make_array((10, 10)) for i in range(4)}
    ref = testee(a)
    res = jace.jit(testee)(a)

    assert len(res) == len(ref)
    assert type(res) == type(ref)
    assert (np.allclose(res[k], ref[k]) for k in ref)
