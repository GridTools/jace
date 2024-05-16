# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the jit decorator.

Also see the `test_jax_api.py` test file, that tests composability.
."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence

import dace
import jax
import numpy as np
from jax import core as jax_core

import jace
from jace import translator


def test_decorator_annotation():
    """Tests the annotation, essential `jace.jax.api_helper.jax_wrapper`."""
    assert jax.jit.__doc__ == jace.jit.__doc__


def test_decorator_individually():
    """Tests the compilation steps individually."""
    jax.config.update("jax_enable_x64", True)

    def testee_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    @jace.jit
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return testee_(A, B)

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    lowered = testee.lower(A, B)
    compiled = lowered.compile()

    ref = testee_(A, B)
    res = compiled(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


def test_decorator_one_go():
    """Tests the compilation steps in one go."""
    jax.config.update("jax_enable_x64", True)

    def testee_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    testee = jace.jit(testee_)

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    ref = testee_(A, B)
    res = testee(A, B)

    assert np.allclose(ref, res), f"Expected '{ref}' got '{res}'."


def test_decorator_caching():
    """This tests the caching ability"""
    jax.config.update("jax_enable_x64", True)

    def testee1_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A * B

    def testee2_(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    testee1 = jace.jit(testee1_)
    testee2 = jace.jit(testee2_)

    assert testee1.__wrapped__ == testee1_
    assert testee2.__wrapped__ == testee2_

    # This is the first size
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # This is the second sizes
    C = np.arange(16, dtype=np.float64).reshape((4, 4))
    D = np.full((4, 4), 10, dtype=np.float64)

    # Lower the two functions for the first size.
    lowered1_size1 = testee1.lower(A, B)
    lowered2_size1 = testee2.lower(A, B)

    # If we now lower them again, we should get the same objects
    assert lowered1_size1 is testee1.lower(A, B)
    assert lowered2_size1 is testee2.lower(A, B)

    # Now we lower them for the second sizes.
    lowered1_size2 = testee1.lower(C, D)
    lowered2_size2 = testee2.lower(C, D)

    # Again if we now lower them again, we should get the same objects.
    assert lowered1_size1 is testee1.lower(A, B)
    assert lowered2_size1 is testee2.lower(A, B)
    assert lowered1_size2 is testee1.lower(C, D)
    assert lowered2_size2 is testee2.lower(C, D)

    # Now use the compilation; since all is the same code path we only use one size.
    compiled1 = lowered1_size1.compile()
    compiled2 = lowered1_size1.compile({"dummy_option": True})

    assert compiled1 is lowered1_size1.compile()
    assert compiled2 is lowered1_size1.compile({"dummy_option": True})
    assert compiled2 is not lowered1_size1.compile({"dummy_option": False})
    assert compiled2 is lowered1_size1.compile({"dummy_option": True})


def test_decorator_double_annot():
    """Tests the behaviour for double annotations."""
    jax.config.update("jax_enable_x64", True)

    lower_cnt = [0, 0]

    def testee1(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        lower_cnt[0] += 1
        return A * B

    def testee2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        lower_cnt[1] += 1
        return A * B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    jaceWrapped1_1 = jace.jit(testee1)
    jaceWrapped1_2 = jace.jit(testee1)
    assert jaceWrapped1_1 is not jaceWrapped1_2

    # Lower them right after the other.
    lower1_1 = jaceWrapped1_1.lower(A, B)
    lower1_2 = jaceWrapped1_2.lower(A, B)
    assert lower1_1 is lower1_2
    assert (
        lower_cnt[0] == 1
    ), f"Annotated right after each other, but lowered {lower_cnt[0]} times instead of once."

    # Now modify the state in between.
    jaceWrapped2_1 = jace.jit(testee2)
    lower2_1 = jaceWrapped2_1.lower(A, B)

    @jace.translator.add_fsubtranslator("non_existing_primitive")
    def non_existing_primitive_translator(
        driver: translator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        raise NotImplementedError

    jaceWrapped2_2 = jace.jit(testee2)
    lower2_2 = jaceWrapped2_2.lower(A, B)
    assert lower2_1 is not lower2_2
    assert lower_cnt[1] == 2

    # Now lower 2_1 again, to see if there is really no influence.
    lower2_1_ = jaceWrapped2_1.lower(A, B)
    assert lower2_1_ is lower2_1
    assert lower_cnt[1] == 2


def test_decorator_sharing():
    """Tests if there is no false sharing in the cache."""
    jax.config.update("jax_enable_x64", True)

    @jace.jit
    def jaceWrapped(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        C = A * B
        D = C + A
        E = D + B  # Just enough state.
        return A + B + C + D + E

    # These are the argument
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # Now we lower it.
    jaceLowered = jaceWrapped.lower(A, B)

    # Now we compile it with enabled optimization.
    optiCompiled = jaceLowered.compile({"auto_optimize": True, "simplify": True})

    # Now we compile it without any optimization.
    unoptiCompiled = jaceLowered.compile({})

    # Because of the way how things work the optimized must have more than the unoptimized.
    #  If there is sharing, then this would not be the case.
    assert optiCompiled._csdfg.sdfg.number_of_nodes() == 1
    assert optiCompiled._csdfg.sdfg.number_of_nodes() < unoptiCompiled._csdfg.sdfg.number_of_nodes()
