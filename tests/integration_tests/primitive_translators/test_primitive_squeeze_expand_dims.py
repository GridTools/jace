# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests about the `squeeze` primitive.

For several reasons parts of the tests related to broadcasting, especially the
ones in which a single dimension is added, are also here. This is because of
the inverse relationship between `expand_dims` and `squeeze`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


if TYPE_CHECKING:
    from collections.abc import Sequence


def _roundtrip_implementation(shape: Sequence[int], axis: int | Sequence[int]) -> None:
    """Implementation of the test for `expand_dims()` and `squeeze()`.

    It will first add dimensions and then remove them.

    Args:
        shape:  Shape of the input array.
        axes:   A series of axis that should be tried.
    """
    a = testutil.make_array(shape)
    a_org = a.copy()

    for ops in [jnp.expand_dims, jnp.squeeze]:
        with jax.experimental.enable_x64():
            ref = ops(a, axis)  # type: ignore[operator]  # Function of unknown type.
            res = jace.jit(lambda a: ops(a, axis))(a)  # type: ignore[operator]  # noqa: B023

        assert ref.shape == res.shape, f"a.shape = {shape}; Expected: {ref.shape}; Got: {res.shape}"
        assert ref.dtype == res.dtype
        assert np.all(ref == res), f"Value error for shape '{shape}' and axis={axis}"
        a = np.array(ref, copy=True)  # It is a JAX array, and we have to reverse this.
    assert a_org.shape == res.shape
    assert np.all(a_org == res)


@pytest.fixture(params=[0, -1, 1])
def single_axis(request) -> int:
    return request.param


@pytest.fixture(params=[0, -1, (1, 2, 3), (3, 2, 1)])
def multiple_axis(request) -> tuple[int, ...] | int:
    return request.param


def test_expand_squeeze_rountrip_simple(single_axis: int) -> None:
    _roundtrip_implementation((10,), single_axis)


def test_expand_squeeze_rountrip_big(multiple_axis: Sequence[int]) -> None:
    _roundtrip_implementation((2, 3, 4, 5), multiple_axis)
