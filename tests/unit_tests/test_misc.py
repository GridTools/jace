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

    Note that this is more or less a test for the calling implementation of
    the `CompiledSDFG` class in DaCe. As I understand the
    `CompiledSDFG::_construct_args()` function this should be detected.
    However, as evidently it does not do this.
    """

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return -A

    # Different types.
    A1 = testutil.mkarray((4, 3), dtype=np.float32)
    A2 = testutil.mkarray((4, 3), dtype=np.int64)

    # Lower and compilation for first type
    callee = testee.lower(A1).compile()

    # But calling with the second type
    with pytest.raises(Exception):  # noqa: B017, PT011 # Unknown exception.
        _ = callee(A2)
