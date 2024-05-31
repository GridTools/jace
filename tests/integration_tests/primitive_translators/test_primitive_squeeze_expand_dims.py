# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for the squeeze translator.

For several reasons parts of the tests related to broadcasting, especially the ones in which
a single dimension is added, are also here. This is because of the inverse relationship between
`expand_dims` and `squeeze`.
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


def _roundtrip_implementation(
    shape: Sequence[int],
    axis: int | Sequence[int],
) -> None:
    """Implementation of the test for `expand_dims()` and `squeeze()`.

    It will first add dimensions and then remove them.

    Args:
        shape:  Shape of the input array.
        axes:   A series of axis that should be tried.
    """
    A = testutil.mkarray(shape)
    A_org = A.copy()

    for ops in [jnp.expand_dims, jnp.squeeze]:
        with jax.experimental.enable_x64():
            ref = ops(A, axis)  # type: ignore[operator]  # Function of unknown type.
            res = jace.jit(lambda A: ops(A, axis))(A)  # type: ignore[operator]  # noqa: B023

        assert ref.shape == res.shape, f"A.shape = {shape}; Expected: {ref.shape}; Got: {res.shape}"
        assert ref.dtype == res.dtype
        assert np.all(ref == res), f"Value error for shape '{shape}' and axis={axis}"
        A = np.array(ref, copy=True)  # It is a Jax array, and we have to reverse this.
    assert A_org.shape == res.shape
    assert np.all(A_org == res)


@pytest.fixture(params=[0, -1, 1])
def simple_axis(request) -> int:
    return request.param


@pytest.fixture(
    params=[
        0,
        -1,
        (1, 2, 3),
        (3, 2, 1),
    ]
)
def hard_axis(request) -> Sequence[int] | int:
    return request.param


def test_expand_squeeze_rountrip_simple(simple_axis):
    _roundtrip_implementation((10,), simple_axis)


def test_expand_squeeze_rountrip_big(hard_axis):
    _roundtrip_implementation((2, 3, 4, 5), hard_axis)
