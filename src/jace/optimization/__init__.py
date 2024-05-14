# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module that will host all optimization functions specific to Jace.

Currently it is just a dummy that exports some functions that do nothing.
"""

from __future__ import annotations

import dace

from jace.translator import post_translation as ptrans


def jace_auto_optimize(
    fsdfg: ptrans.FinalizedJaxprSDFG,
    simplify: bool = True,
    **kwargs: str | bool,  # noqa: ARG001  # Unused argument, for now
) -> dace.SDFG:
    """Performs optimization of the `fsdfg` _inplace_ and returns it.

    Currently this function only supports simplification.
    Its main job is to exists that we have something that we can call in the tool chain.
    """
    if simplify:
        fsdfg.sdfg.simplify()

    fsdfg.validate()
    return fsdfg


__all__ = [
    "jace_auto_optimize",
]
