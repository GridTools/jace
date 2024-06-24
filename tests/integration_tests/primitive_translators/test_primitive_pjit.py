# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the `pjit` primitive."""

from __future__ import annotations

from collections.abc import Generator

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


@pytest.fixture(autouse=True)
def _disable_jit() -> Generator[None, None, None]:
    """Overwrites the global `_disable_jit` fixture and enables jit operations."""
    with jax.disable_jit(disable=False):
        yield


def test_pjit_simple() -> None:
    """Simple nested Jaxpr expression."""

    def testee(a: np.ndarray) -> np.ndarray:
        return jax.jit(lambda a: jnp.sin(a))(a)  # noqa: PLW0108 [unnecessary-lambda]  # Lambda needed to trigger a `pjit` level.

    a = testutil.make_array((10, 10))

    jace_wrapped = jace.jit(testee)
    jace_lowered = jace_wrapped.lower(a)
    res = jace_wrapped(a)
    ref = testee(a)

    assert jace_lowered._jaxpr.eqns[0].primitive.name == "pjit"
    assert np.allclose(res, ref)
    assert res.dtype == ref.dtype
    assert res.shape == ref.shape


@pytest.mark.parametrize("shape", [(10, 10), ()])
def test_pjit_literal(shape) -> None:
    """Test for `pjit` with literal inputs."""

    def testee(pred: np.ndarray, fbranch: np.ndarray) -> jax.Array:
        return jnp.where(pred, 2, fbranch)

    pred = testutil.make_array(shape, np.bool_)
    fbranch = pred * 0

    jace_wrapped = jace.jit(testee)
    jace_lowered = jace_wrapped.lower(pred, fbranch)
    res = jace_wrapped(pred, fbranch)
    ref = testee(pred, fbranch)

    assert np.all(ref == res)
    assert jace_lowered._jaxpr.eqns[0].primitive.name == "pjit"
    assert any(isinstance(invar, jax.core.Literal) for invar in jace_lowered._jaxpr.eqns[0].invars)
    assert res.dtype == ref.dtype
