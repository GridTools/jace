# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the caching infrastructure.
."""

from __future__ import annotations

import itertools as it
import re

import numpy as np
import pytest

import jace
from jace import optimization
from jace.jax import stages


@pytest.fixture(autouse=True)
def _clear_translation_cache():
    """Decorator that clears the translation cache.

    Ensures that a function finds an empty cache and clears up afterwards.

    Todo:
        Ask Enrique how I can make that fixture apply everywhere not just in the file but the whole test suite.
    """
    from jace.jax import translation_cache as tcache

    tcache.clear_translation_cache()
    yield
    tcache.clear_translation_cache()


def test_caching_same_sizes():
    """The behaviour of the cache if same sizes are used."""

    # Counter for how many time it was lowered.
    lowering_cnt = [0]

    # This is the pure Python function.
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A * B

    # this is the wrapped function.
    @jace.jit
    def wrapped(A, B):
        lowering_cnt[0] += 1
        return testee(A, B)

    # First batch of arguments.
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # The second batch of argument, it is the same size (structurally) but different values.
    AA = A + 1.0362
    BB = B + 0.638956

    # Now let's lower it once directly and call it.
    lowered: stages.JaceLowered = wrapped.lower(A, B)
    compiled: stages.JaceCompiled = lowered.compile()
    assert lowering_cnt[0] == 1
    assert np.allclose(testee(A, B), compiled(A, B))

    # Now lets call the wrapped object directly, since we already did the lowering
    #  no longering (and compiling) is needed.
    assert np.allclose(testee(A, B), wrapped(A, B))
    assert lowering_cnt[0] == 1

    # Now lets call it with different objects, that have the same structure.
    #  Again no lowering should happen.
    assert np.allclose(testee(AA, BB), wrapped(AA, BB))
    assert wrapped.lower(AA, BB) is lowered
    assert wrapped.lower(A, B) is lowered
    assert lowering_cnt[0] == 1


def test_caching_different_sizes():
    """The behaviour of the cache if different sizes where used."""

    # Counter for how many time it was lowered.
    lowering_cnt = [0]

    # This is the wrapped function.
    @jace.jit
    def wrapped(A, B):
        lowering_cnt[0] += 1
        return A * B

    # First size of arguments
    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    # Second size of arguments
    C = np.arange(16, dtype=np.float64).reshape((4, 4))
    D = np.full((4, 4), 10, dtype=np.float64)

    # Now lower the function once for each.
    lowered1 = wrapped.lower(A, B)
    lowered2 = wrapped.lower(C, D)
    assert lowering_cnt[0] == 2
    assert lowered1 is not lowered2

    # Now also check if the compilation works as intended
    compiled1 = lowered1.compile()
    compiled2 = lowered2.compile()
    assert lowering_cnt[0] == 2
    assert compiled1 is not compiled2


@pytest.mark.skip(reason="Missing primitive translators")
def test_caching_different_structure():
    """Now tests if we can handle multiple arguments with different structures.

    Todo:
        - Extend with strides once they are part of the cache.
    """

    # This is the wrapped function.
    lowering_cnt = [0]

    @jace.jit
    def wrapped(A, B):
        lowering_cnt[0] += 1
        return A * 4.0, B + 2.0

    A = np.full((4, 30), 10, dtype=np.float64)
    B = np.full((4, 3), 10, dtype=np.float64)
    C = np.full((5, 3), 14, dtype=np.float64)
    D = np.full((6, 3), 14, dtype=np.int64)

    # These are the arrays.
    args: dict[int, np.ndarray] = {id(x): x for x in [A, B, C, D]}
    # These are the known lowerings.
    lowerings: dict[tuple[int, int], stages.JaceLowered] = {}
    lowering_ids: set[int] = set()
    # These are the known compilations.
    compilations: dict[tuple[int, int], stages.JaceCompiled] = {}
    compiled_ids: set[int] = set()

    # Generating the lowerings
    for arg1, arg2 in it.permutations([A, B, C, D], 2):
        lower = wrapped.lower(arg1, arg2)
        compiled = lower.compile()
        assert id(lower) not in lowering_ids
        assert id(compiled) not in compiled_ids
        lowerings[id(arg1), id(arg2)] = lower
        lowering_ids.add(id(lower))
        compilations[id(arg1), id(arg2)] = compiled
        compiled_ids.add(id(compiled))

    # Now check if they are still cached.
    for arg1, arg2 in it.permutations([A, B, C, D], 2):
        lower = wrapped.lower(arg1, arg2)
        clower = lowerings[id(arg1), id(arg2)]
        assert clower is lower

        compiled1 = lower.compile()
        compiled2 = clower.compile()
        ccompiled = compilations[id(arg1), id(arg2)]
        assert compiled1 is compiled2
        assert compiled1 is ccompiled


def test_caching_compilation():
    """Tests the compilation cache, this is just very simple, since it uses the same code paths as lowering."""

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

    # Compiling it without any information.
    optiCompiled = jaceLowered.compile()

    # This should be the same as passing the defaults directly.
    assert optiCompiled is jaceLowered.compile(optimization.DEFAULT_OPTIMIZATIONS)

    # Also if we pass the empty dict, we should get the default.
    assert optiCompiled is jaceLowered.compile({})

    # Now we disable all optimizations
    unoptiCompiled = jaceLowered.compile(optimization.NO_OPTIMIZATIONS)

    # Because of the way how things work the optimized must have more than the unoptimized.
    #  If there is sharing, then this would not be the case.
    assert unoptiCompiled is not optiCompiled
    assert optiCompiled._csdfg.sdfg.number_of_nodes() == 1
    assert optiCompiled._csdfg.sdfg.number_of_nodes() < unoptiCompiled._csdfg.sdfg.number_of_nodes()


def test_caching_dtype():
    """Tests if the data type is properly included in the test."""

    lowering_cnt = [0]

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        lowering_cnt[0] += 1
        return A + A

    dtypes = [np.float64, np.float32, np.int32, np.int64]
    shape = (10, 10)

    for i, dtype in enumerate(dtypes):
        A = np.array((np.random.random(shape) - 0.5) * 10, dtype=dtype)  # noqa: NPY002

        assert lowering_cnt[0] == i
        _ = testee(A)
        assert lowering_cnt[0] == i + 1

        assert np.allclose(testee(A), 2 * A)
        assert lowering_cnt[0] == i + 1


def test_caching_strides() -> None:
    """Test if the cache detects a change in strides."""

    @jace.jit
    def wrapped(A: np.ndarray) -> np.ndarray:
        return A + 10.0

    shape = (10, 100, 1000)
    C = np.array(
        (np.random.random(shape) - 0.5) * 10,  # noqa: NPY002
        order="C",
        dtype=np.float64,
    )
    F = np.array(C, copy=True, order="F")

    # First we compile run it with C strides.
    C_lower = wrapped.lower(C)
    C_res = wrapped(C)

    # Now we run it with FORTRAN strides.
    #  However, this does not work because we do not support strides at all.
    #  But the cache is aware of this, which helps catch some nasty bugs.
    F_lower = None  # Remove later
    F_res = C_res.copy()  # Remove later
    with pytest.raises(  # noqa: PT012 # Multiple calls
        expected_exception=NotImplementedError,
        match=re.escape("Currently can not handle strides beside 'C_CONTIGUOUS'."),
    ):
        F_lower = wrapped.lower(F)
        F_res = wrapped(F)
    assert F_lower is None  # Remove later.
    assert C_res is not F_res  # Remove later
    assert np.allclose(F_res, C_res)
    assert F_lower is not C_lower
