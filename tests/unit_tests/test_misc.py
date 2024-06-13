# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements general tests for JaCe."""

from __future__ import annotations

import numpy as np
import pytest

import jace

from tests import util as testutil


@pytest.mark.skip("Possible bug in DaCe.")
def test_mismatch_in_datatype_calling() -> None:
    """Tests compilation and calling with different types.

    Note that this more or less tests the calling implementation of the `CompiledSDFG`
    class in DaCe. As I understand the `CompiledSDFG::_construct_args()` function this
    should be detected. However, as evidently it does not do this.
    """

    @jace.jit
    def testee(a: np.ndarray) -> np.ndarray:
        return -a

    # Different types.
    a1 = testutil.make_array((4, 3), dtype=np.float32)
    a2 = testutil.make_array((4, 3), dtype=np.int64)

    # Lower and compilation for first type
    callee = testee.lower(a1).compile()

    # But calling with the second type
    with pytest.raises(Exception):  # noqa: B017, PT011 # Unknown exception.
        _ = callee(a2)
