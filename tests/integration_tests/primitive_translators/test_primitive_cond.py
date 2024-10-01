# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace
from jace.util import translation_cache as tcache

from tests import util as testutil


def _perform_cond_test(
    testee: Callable[[np.float64, tuple[Any, ...]], Any], branch_args: tuple[Any, ...]
) -> None:
    """
    Performs a test for the condition primitives.

    It assumes that the first argument is used for the condition and that the
    conditions is applied at `0.5`.
    The test function adds a prologue, that performs some operations on the
    `branch_args` and performs some computations on the final value.
    This is done to simulate the typical usage, as it was observed that
    sometimes the optimization fails.
    """
    tcache.clear_translation_cache()

    def prologue(branch_args: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            jnp.exp(jnp.cos(jnp.sin(branch_arg))) ** i
            for i, branch_arg in enumerate(branch_args, 2)
        )

    def epilogue(result: Any) -> Any:
        return jnp.exp(jnp.sin(jnp.sin(result)))

    def final_testee(
        val: np.float64,
        branch_args: tuple[Any, ...],
    ) -> Any:
        return epilogue(testee(jnp.sin(val) + 0.5, prologue(branch_args)))  # type: ignore[arg-type]

    vals: list[np.float64] = [np.float64(-0.5), np.float64(0.6)]
    wrapped = jace.jit(testee)

    for val in vals:
        res = wrapped(val, branch_args)
        ref = testee(val, branch_args)

        assert np.all(res == ref)
        assert (1,) if ref.shape == () else ref.shape == res.shape


def test_cond_full_branches() -> None:
    def testee(val: np.float64, branch_args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return jax.lax.cond(
            val < 0.5,
            lambda arg: jnp.sin(arg[0]),
            lambda arg: jnp.cos(arg[1]),
            branch_args,
        )

    branch_args = tuple(testutil.make_array(1) for _ in range(2))
    _perform_cond_test(testee, branch_args)


def test_cond_scalar_brnaches() -> None:
    def testee(val: np.float64, branch_args: tuple[np.float64, np.float64]) -> np.float64:
        return jax.lax.cond(
            val < 0.5,
            lambda arg: arg[0] + 2.0,
            lambda arg: arg[1] + 3.0,
            branch_args,
        )

    branch_args = tuple(testutil.make_array(()) for _ in range(2))
    _perform_cond_test(testee, branch_args)


def test_cond_literal_bool() -> None:
    for branch_sel in [True, False]:

        def testee(val: np.float64, branch_args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            return jax.lax.cond(
                branch_sel,  # noqa: B023 [function-uses-loop-variable]
                lambda arg: jnp.sin(arg[0]) + val,
                lambda arg: jnp.cos(arg[1]),
                branch_args,
            )

        branch_args = tuple(testutil.make_array(1) for _ in range(2))
        _perform_cond_test(testee, branch_args)


def test_cond_one_empty_branch() -> None:
    def testee(val, branch_args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return jax.lax.cond(
            val < 0.5,
            lambda xtrue: xtrue[0],
            lambda xfalse: jnp.array([1]) + xfalse[1],
            branch_args,
        )

    branch_args = tuple(testutil.make_array(1) for _ in range(2))
    _perform_cond_test(testee, branch_args)


@pytest.mark.skip(reason="Literal return value is not implemented.")
def test_cond_literal_branch() -> None:
    def testee(val: np.float64, branch_args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return jax.lax.cond(
            val < 0.5,
            lambda xtrue: 1.0,  # noqa: ARG005 [unused-lambda-argument]
            lambda xfalse: xfalse[1],
            branch_args,
        )

    branch_args = tuple(testutil.make_array(()) for _ in range(2))
    _perform_cond_test(testee, branch_args)


def test_cond_complex_branches() -> None:
    def true_branch(arg: np.ndarray) -> np.ndarray:
        return jnp.where(
            jnp.asin(arg) <= 0.0,
            jnp.exp(jnp.cos(jnp.sin(arg))),
            arg * 4.0,
        )

    def false_branch(arg: np.ndarray) -> np.ndarray:
        return true_branch(jnp.exp(jnp.cos(arg) ** 7))  # type: ignore[arg-type]

    def testee(val: np.float64, branch_args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        cond_res = jax.lax.cond(
            val < 0.5,
            lambda arg: true_branch(arg[0]),
            lambda arg: false_branch(arg[1]),
            branch_args,
        )
        return true_branch(cond_res)

    branch_args = tuple(testutil.make_array((100, 100)) for _ in range(2))
    _perform_cond_test(testee, branch_args)


def test_cond_switch() -> None:
    def testee(
        selector: int,
        branch_args: tuple[Any, ...],
    ) -> np.ndarray:
        return jax.lax.switch(
            selector,
            (
                lambda args: jnp.sin(args[0]),
                lambda args: jnp.exp(args[1]),
                lambda args: jnp.cos(args[2]),
            ),
            branch_args,
        )

    wrapped = jace.jit(testee)
    branch_args = tuple(testutil.make_array((100, 100)) for _ in range(3))

    # These are the values that we will use for the selector.
    #  Note that we also use some invalid values.
    selectors = [-1, 0, 1, 2, 3, 4]

    for selector in selectors:
        ref = testee(selector, branch_args)
        res = wrapped(selector, branch_args)

        assert ref.shape == res.shape
        assert np.allclose(ref, res)


@pytest.mark.skip("DaCe is not able to optimize it away.")
def test_cond_switch_literal_selector() -> None:
    def testee(
        branch_args: tuple[Any, ...],
    ) -> np.ndarray:
        return jax.lax.switch(
            2,
            (
                lambda args: jnp.sin(args[0]),
                lambda args: jnp.exp(args[1]),
                lambda args: jnp.cos(args[2]),
            ),
            branch_args,
        )

    branch_args = tuple(testutil.make_array((100, 100)) for _ in range(3))

    wrapped = jace.jit(testee)
    lowered = wrapped.lower(branch_args)
    compiled = lowered.compile(jace.optimization.DEFAULT_OPTIMIZATIONS)

    ref = testee(branch_args)
    res = wrapped(branch_args)

    assert ref.shape == res.shape
    assert np.allclose(ref, res)
    lowered.as_sdfg().view()
    assert compiled._compiled_sdfg.sdfg.number_of_nodes() == 1
