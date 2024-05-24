# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the element type conversion functionality."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pytest
from jax import numpy as jnp

import jace


# fmt: off
_DACE_TYPES: Final[list[type]] = [
        np.int_, np.int8, np.int16, np.int32, np.int64,
        np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
        np.float64, np.float32, np.float64,
]
_DACE_COMPLEX: Final[list[type]] = [
        np.complex128, np.complex64, np.complex128,
]
# fmt: on


def _test_convert_element_type_impl(
    input_types: Sequence,
    output_types: Sequence,
) -> bool:
    """Implementation of the tests of the convert element types primitive."""
    lowering_cnt = [0, 0]
    for input_type in input_types:
        for output_type in output_types:
            A = np.array(np.random.random((10, 10)), dtype=input_type)  # noqa: NPY002
            ref = np.array(A, copy=True, dtype=output_type)
            lowering_cnt[1] += 1

            @jace.jit
            def converter(A: np.ndarray) -> np.ndarray:
                lowering_cnt[0] += 1
                return jnp.array(A, copy=False, dtype=output_type)  # noqa: B023  # Loop variable.

            res = converter(A)
            assert res.dtype == output_type
            assert lowering_cnt[0] == lowering_cnt[1]
            assert np.allclose(ref, res)
    return True


@pytest.mark.skip(reason="Too slow, find way to run only on demand.")
def test_convert_element_type_main():
    """Tests all conversions with the exception of conversions from bool and complex."""
    _test_convert_element_type_impl(_DACE_TYPES, [*_DACE_TYPES, np.bool_])


def test_convert_element_type_main_short():
    """Fast running version of `test_convert_element_type_main()`."""
    FAST_TYPES = [np.int32, np.int64, np.float64]
    _test_convert_element_type_impl(FAST_TYPES, [*FAST_TYPES, np.bool_])


def test_convert_element_type_complex():
    """All complex conversions."""
    _test_convert_element_type_impl(_DACE_COMPLEX, _DACE_COMPLEX)


def test_convert_element_type_from_bool():
    """Tests conversions from bools to any other types."""
    _test_convert_element_type_impl([np.bool_], _DACE_COMPLEX)


def test_convert_element_type_useless_cast():
    """Broadcast a literal to a matrix.

    This test is here to show, that in certain situation Jax inserts
    a `convert_element_type` primitive even if it is not needed.
    """

    def testee(a: float) -> np.ndarray:
        # For it to work we have to use `numpy` instead of the Jax substitute.
        return np.broadcast_to(1.0, (10, 10)) + a

    with pytest.warns(
        expected_warning=UserWarning,
        match=r"convert_element_type\(.*\): is useless, because input and output have same type.",
    ):
        res = jace.jit(testee)(1.0)

    ref = testee(1.0)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert np.all(res == ref)
