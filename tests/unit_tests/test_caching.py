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

import jax
import numpy as np
from jax import numpy as jnp

import jace
from jace import optimization, stages
from jace.util import translation_cache as tcache

from tests import util as testutil


def test_caching_working() -> None:
    """Simple test if the caching actually works."""

    lowering_cnt = [0]

    @jace.jit
    def wrapped(a: np.ndarray) -> jax.Array:
        lowering_cnt[0] += 1
        return jnp.sin(a)

    a = testutil.make_array((10, 10))
    ref = np.sin(a)
    res_ids: set[int] = set()
    # We have to store the array, because numpy does reuse the memory.
    res_set: list[jax.Array] = []

    for _ in range(10):
        res = wrapped(a)
        res_id = res.__array_interface__["data"][0]  # type: ignore[attr-defined]

        assert np.allclose(res, ref)
        assert lowering_cnt[0] == 1
        assert res_id not in res_ids
        res_ids.add(res_id)
        res_set.append(res)


def test_caching_same_sizes() -> None:
    """The behaviour of the cache if same sizes are used, in two different functions."""

    # Counter for how many time it was lowered.
    lowering_cnt = [0]

    # This is the pure Python function.
    def testee(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    # this is the wrapped function.
    @jace.jit
    def wrapped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lowering_cnt[0] += 1
        return testee(a, b)

    # First batch of arguments.
    a = testutil.make_array((4, 3))
    b = testutil.make_array((4, 3))

    # The second batch of argument, same structure, but different values.
    aa = a + 1.0362
    bb = b + 0.638956

    # Now let's lower it once directly and call it.
    lowered: stages.JaCeLowered = wrapped.lower(a, b)
    compiled: stages.JaCeCompiled = lowered.compile()
    assert lowering_cnt[0] == 1
    assert np.allclose(testee(a, b), compiled(a, b))

    # Now lets call the wrapped object directly, since we already did the lowering
    #  no lowering (and compiling) is needed.
    assert np.allclose(testee(a, b), wrapped(a, b))
    assert lowering_cnt[0] == 1

    # Now lets call it with different objects, that have the same structure.
    #  Again no lowering should happen.
    assert np.allclose(testee(aa, bb), wrapped(aa, bb))
    assert wrapped.lower(aa, bb) is lowered
    assert wrapped.lower(a, b) is lowered
    assert lowering_cnt[0] == 1


def test_caching_different_sizes() -> None:
    """The behaviour of the cache if different sizes where used."""

    # Counter for how many time it was lowered.
    lowering_cnt = [0]

    # This is the wrapped function.
    @jace.jit
    def wrapped(a, b):
        lowering_cnt[0] += 1
        return a * b

    # First size of arguments
    a = testutil.make_array((4, 3))
    b = testutil.make_array((4, 3))

    # Second size of arguments
    c = testutil.make_array((4, 4))
    d = testutil.make_array((4, 4))

    # Now lower the function once for each.
    lowered1 = wrapped.lower(a, b)
    lowered2 = wrapped.lower(c, d)
    assert lowering_cnt[0] == 2
    assert lowered1 is not lowered2

    # Now also check if the compilation works as intended
    compiled1 = lowered1.compile()
    compiled2 = lowered2.compile()
    assert lowering_cnt[0] == 2
    assert compiled1 is not compiled2


def test_caching_different_structure() -> None:
    """Now tests if we can handle multiple arguments with different structures.

    Todo:
        - Extend with strides once they are part of the cache.
    """

    # This is the wrapped function.
    lowering_cnt = [0]

    @jace.jit
    def wrapped(a, b):
        lowering_cnt[0] += 1
        return a * 4.0, b + 2.0

    a = testutil.make_array((4, 30), dtype=np.float64)
    b = testutil.make_array((4, 3), dtype=np.float64)
    c = testutil.make_array((4, 3), dtype=np.int64)
    d = testutil.make_array((6, 3), dtype=np.int64)

    # These are the known lowered instances.
    lowerings: dict[tuple[int, int], stages.JaCeLowered] = {}
    lowering_ids: set[int] = set()
    # These are the known compilation instances.
    compilations: dict[tuple[int, int], stages.JaCeCompiled] = {}
    compiled_ids: set[int] = set()

    # Generating the lowerings
    for arg1, arg2 in it.permutations([a, b, c, d], 2):
        lower = wrapped.lower(arg1, arg2)
        compiled = lower.compile()
        assert id(lower) not in lowering_ids
        assert id(compiled) not in compiled_ids
        lowerings[id(arg1), id(arg2)] = lower
        lowering_ids.add(id(lower))
        compilations[id(arg1), id(arg2)] = compiled
        compiled_ids.add(id(compiled))

    # Now check if they are still cached.
    for arg1, arg2 in it.permutations([a, b, c, d], 2):
        lower = wrapped.lower(arg1, arg2)
        clower = lowerings[id(arg1), id(arg2)]
        assert clower is lower

        compiled1 = lower.compile()
        compiled2 = clower.compile()
        ccompiled = compilations[id(arg1), id(arg2)]
        assert compiled1 is compiled2
        assert compiled1 is ccompiled


def test_caching_compilation() -> None:
    """Tests the compilation cache."""

    @jace.jit
    def jace_wrapped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        c = a * b
        d = c + a
        e = d + b  # Just enough state.
        return a + b + c + d + e

    # These are the argument
    a = testutil.make_array((4, 3))
    b = testutil.make_array((4, 3))

    # Now we lower it.
    jace_lowered = jace_wrapped.lower(a, b)

    # Compiling it with and without optimizations enabled
    optized_compiled = jace_lowered.compile(optimization.DEFAULT_OPTIMIZATIONS)
    unoptized_compiled = jace_lowered.compile(optimization.NO_OPTIMIZATIONS)

    # Because of the way how things work the optimized must have more than the
    #  unoptimized. If there is sharing, then this would not be the case.
    assert unoptized_compiled is not optized_compiled
    assert optized_compiled._compiled_sdfg.sdfg.number_of_nodes() == 1
    assert (
        optized_compiled._compiled_sdfg.sdfg.number_of_nodes()
        < unoptized_compiled._compiled_sdfg.sdfg.number_of_nodes()
    )

    # Now we check if they are still inside the cache.
    assert optized_compiled is jace_lowered.compile(optimization.DEFAULT_OPTIMIZATIONS)
    assert unoptized_compiled is jace_lowered.compile(optimization.NO_OPTIMIZATIONS)


def test_caching_compilation_options() -> None:
    """Tests if the global optimization managing works."""
    original_compile_options = stages.get_active_compiler_options()
    try:
        lowering_cnt = [0]

        @jace.jit
        def wrapped(a: float) -> float:
            lowering_cnt[0] += 1
            return a + 1.0

        lower_cache = wrapped._cache
        lowered = wrapped.lower(1.0)
        compile_cache = lowered._cache

        assert len(lower_cache) == 1
        assert len(compile_cache) == 0
        assert lowering_cnt[0] == 1

        # Using the first set of options.
        stages.update_active_compiler_options(optimization.NO_OPTIMIZATIONS)
        _ = wrapped(2.0)

        # Except from one entry in the compile cache, nothing should have changed.
        assert len(lower_cache) == 1
        assert len(compile_cache) == 1
        assert compile_cache.front()[0].stage_id == id(lowered)
        assert lowering_cnt[0] == 1

        # Now we change the options again which then will lead to another compilation,
        #  but not to another lowering.
        stages.update_active_compiler_options(optimization.DEFAULT_OPTIMIZATIONS)
        _ = wrapped(2.0)

        assert len(lower_cache) == 1
        assert len(compile_cache) == 2
        assert compile_cache.front()[0].stage_id == id(lowered)
        assert lowering_cnt[0] == 1

    finally:
        stages.update_active_compiler_options(original_compile_options)


def test_caching_dtype() -> None:
    """Tests if the data type is properly included in the test."""

    lowering_cnt = [0]

    @jace.jit
    def testee(a: np.ndarray) -> np.ndarray:
        lowering_cnt[0] += 1
        return a + a

    dtypes = [np.float64, np.float32, np.int32, np.int64]
    shape = (10, 10)

    for i, dtype in enumerate(dtypes):
        a = testutil.make_array(shape, dtype=dtype)

        # First lowering
        assert lowering_cnt[0] == i
        _ = testee(a)
        assert lowering_cnt[0] == i + 1

        # Second, implicit, lowering, which must be cached.
        assert np.allclose(testee(a), 2 * a)
        assert lowering_cnt[0] == i + 1


def test_caching_eviction_simple() -> None:
    """Simple tests for cache eviction."""

    @jace.jit
    def testee(a: np.ndarray) -> np.ndarray:
        return a + 1.0

    cache: tcache.StageCache = testee._cache
    assert len(cache) == 0

    first_lowered = testee.lower(np.ones(10))
    first_key = cache.front()[0]
    assert len(cache) == 1

    second_lowered = testee.lower(np.ones(11))
    second_key = cache.front()[0]
    assert len(cache) == 2
    assert second_key != first_key

    third_lowered = testee.lower(np.ones(12))
    third_key = cache.front()[0]
    assert len(cache) == 3
    assert third_key != second_key
    assert third_key != first_key

    # Test if the key association is correct.
    #  We have to do it in this order, because reading the key modifies the order.
    assert cache.front()[0] == third_key
    assert cache[first_key] is first_lowered
    assert cache.front()[0] == first_key
    assert cache[second_key] is second_lowered
    assert cache.front()[0] == second_key
    assert cache[third_key] is third_lowered
    assert cache.front()[0] == third_key

    # We now evict the second key, which should not change anything on the order.
    cache.popitem(second_key)
    assert len(cache) == 2
    assert first_key in cache
    assert second_key not in cache
    assert third_key in cache
    assert cache.front()[0] == third_key

    # Now we modify first_key, which moves it to the front.
    cache[first_key] = first_lowered
    assert len(cache) == 2
    assert first_key in cache
    assert third_key in cache
    assert cache.front()[0] == first_key

    # Now we evict the oldest one, which is third_key
    cache.popitem(None)
    assert len(cache) == 1
    assert first_key in cache
    assert cache.front()[0] == first_key


def test_caching_eviction_complex() -> None:
    """Tests if the stuff is properly evicted if the cache is full."""

    @jace.jit
    def testee(a: np.ndarray) -> np.ndarray:
        return a + 1.0

    cache: tcache.StageCache = testee._cache
    capacity = cache.capacity
    assert len(cache) == 0

    # Lets fill the cache to the brim.
    for i in range(capacity):
        a = np.ones(i + 10)
        lowered = testee.lower(a)
        assert len(cache) == i + 1

        if i == 0:
            first_key: tcache.StageTransformationSpec = cache.front()[0]
            first_lowered = cache[first_key]
            assert lowered is first_lowered
        elif i == 1:
            second_key: tcache.StageTransformationSpec = cache.front()[0]
            assert second_key != first_key
            assert cache[second_key] is lowered
        assert first_key in cache

    assert len(cache) == capacity
    assert first_key in cache
    assert second_key in cache

    # Now we will modify the first key, this should make it the newest.
    assert cache.front()[0] != first_key
    cache[first_key] = first_lowered
    assert len(cache) == capacity
    assert first_key in cache
    assert second_key in cache
    assert cache.front()[0] == first_key

    # Now we will add a new entry to the cache, this will evict the second entry.
    _ = testee.lower(np.ones(capacity + 1000))
    assert len(cache) == capacity
    assert cache.front()[0] != first_key
    assert first_key in cache
    assert second_key not in cache


def test_caching_strides() -> None:
    """Test if the cache detects a change in strides."""

    lower_cnt = [0]

    @jace.jit
    def wrapped(a: np.ndarray) -> np.ndarray:
        lower_cnt[0] += 1
        return a + 10.0

    shape = (10, 100, 1000)
    array_c = testutil.make_array(shape, order="c")
    array_f = np.array(array_c, copy=True, order="F")

    # First we compile run it with c strides.
    lower_c = wrapped.lower(array_c)
    res_c = wrapped(array_c)

    lower_f = wrapped.lower(array_f)
    res_f = lower_f.compile()(array_f)

    assert res_c is not res_f
    assert np.allclose(res_f, res_c)
    assert lower_f is not lower_c
    assert lower_cnt[0] == 2


def test_caching_jax_numpy_array() -> None:
    """Tests if jax arrays are handled the same way as numpy array."""

    def _test_impl(
        for_lowering: np.ndarray | jax.Array, for_calling: np.ndarray | jax.Array
    ) -> None:
        tcache.clear_translation_cache()
        lowering_cnt = [0]

        @jace.jit
        def wrapped(a: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
            lowering_cnt[0] += 1
            return a + 1.0

        # Explicit lowering.
        _ = wrapped(for_lowering)
        assert lowering_cnt[0] == 1

        # Now calling with the second argument, it should not longer again.
        _ = wrapped(for_calling)
        assert lowering_cnt[0] == 1, "Expected no further lowering."

    a_numpy = testutil.make_array((10, 10))
    a_jax = jnp.array(a_numpy, copy=True)
    assert a_numpy.dtype == a_jax.dtype

    _test_impl(a_numpy, a_jax)
    _test_impl(a_jax, a_numpy)
