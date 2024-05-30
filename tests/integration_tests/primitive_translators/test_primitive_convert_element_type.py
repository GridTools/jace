# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the element type conversion functionality."""

from __future__ import annotations

from typing import Final

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace
from jace.util import translation_cache as tcache


# fmt: off
_DACE_REAL_TYPES: Final[list[type]] = [
        np.int_, np.int8, np.int16, np.int32, np.int64,
        np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
        np.float64, np.float32, np.float64,
]
_DACE_COMPLEX_TYPES: Final[list[type]] = [
        np.complex128, np.complex64, np.complex128,
]
# fmt: on


@pytest.fixture(autouse=True)
def _clear_translation_cache():
    """Decorator that clears the translation cache.

    Ensures that a function finds an empty cache and clears up afterwards.

    Todo:
        Ask Enrique how I can make that fixture apply everywhere not just in the file but the whole test suite.
    """
    tcache.clear_translation_cache()
    yield
    tcache.clear_translation_cache()


@pytest.fixture(params=_DACE_REAL_TYPES)
def src_type(request) -> type:
    """All valid source types, with the exception of bool."""
    return request.param


@pytest.fixture(params=_DACE_REAL_TYPES + _DACE_COMPLEX_TYPES)
def dst_type(request) -> type:
    """All valid destination types, with the exception of bool.

    Includes also complex types, because going from real to complex is useful, but the other
    way is not.
    """
    return request.param


def _convert_element_type_impl(
    input_type: type,
    output_type: type,
) -> bool:
    """Implementation of the tests of the convert element types primitive."""
    lowering_cnt = [0]
    A: np.ndarray = np.array(np.random.random((10, 10)), dtype=input_type)  # noqa: NPY002
    assert A.dtype == input_type
    ref: np.ndarray = np.array(A, copy=True, dtype=output_type)
    assert ref.dtype == output_type

    @jace.jit
    def converter(A: np.ndarray) -> jax.Array:
        lowering_cnt[0] += 1
        return jnp.array(A, copy=False, dtype=output_type)  # Loop variable.

    res = converter(A)
    assert lowering_cnt[0] == 1
    assert (
        res.dtype == output_type
    ), f"Expected '{output_type}', but got '{res.dtype}', input was '{input_type}'."
    assert np.allclose(ref, res)
    return True


@pytest.mark.skip(reason="This test is too long, only do it on certain conditions.")
def test_convert_element_type_main(src_type, dst_type):
    """Tests all conversions with the exception of conversions from bool and complex."""
    _convert_element_type_impl(src_type, dst_type)


@pytest.mark.skip(reason="This test is too long, only do it on certain conditions.")
def test_convert_element_type_from_bool(src_type):
    _convert_element_type_impl(np.bool_, src_type)


@pytest.mark.skip(reason="This test is too long, only do it on certain conditions.")
def test_convert_element_type_to_bool(dst_type):
    _convert_element_type_impl(dst_type, np.bool_)


@pytest.mark.skip(reason="The warning was disabled, so the test is at the moment useless.")
def test_convert_element_type_useless_cast():
    """Shows that under some conditions there is really a casting from one type to the same.

    In certain cases, also in some slicing tests, this useless cast is inserted by Jax.
    This test was originally here to show this. However, that thing got so annoying that it was
    removed. The test is kept here to serve as some kind of a reference.
    """

    def testee(a: float) -> np.ndarray:
        # For it to work we have to use `numpy` instead of the Jax substitute.
        return np.broadcast_to(1.0, (10, 10)) + a

    with pytest.warns(
        expected_warning=UserWarning,
        match=r"convert_element_type\(.*\): is useless, input and output have same type\.",
    ):
        res = jace.jit(testee)(1.0)

    ref = testee(1.0)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert np.all(res == ref)
