# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module that will host all optimization functions specific to Jace.

Currently just a dummy existing for the sake of providing some callable function.
"""

from __future__ import annotations

from jace import translator


def jace_optimize(
    tsdfg: translator.TranslatedJaxprSDFG,
    simplify: bool = True,
    auto_optimize: bool = False,
) -> None:
    """Performs optimization of the `fsdfg` _inplace_.

    Currently this function only supports simplification.
    Its main job is to exists that we have something that we can call in the tool chain.

    Args:
        simplify:       Run the simplification pilepline.
        auto_optimize:  Run the auto optimization pipeline (currently does nothing)
    """
    if not tsdfg.is_finalized:
        raise ValueError("Can only optimize finalized SDFGs.")

    if simplify:
        tsdfg.sdfg.simplify()

    if auto_optimize:
        pass

    tsdfg.validate()
