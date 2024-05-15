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

from jace import translator


def jace_optimize(
    tsdfg: translator.TranslatedJaxprSDFG,
    simplify: bool = False,
    auto_optimize: bool = False,
    **kwargs: str | bool,  # noqa: ARG001  # Unused argument, for now
) -> None:
    """Performs optimization of the `fsdfg` _inplace_.

    Currently this function only supports simplification.
    Its main job is to exists that we have something that we can call in the tool chain.

    Args:
        simplify:       Run the simplification pilepline.
        auto_optimize:  Run the auto optimization pipeline (currently does nothing)

    Notes:
        All optimization flags must be disabled by default!
            The reason for this is that `jaceLowered.compile({})` will disable all optimizations.
    """
    if not tsdfg.is_finalized:
        raise ValueError("Can only optimize finalized SDFGs.")

    if simplify:
        tsdfg.sdfg.simplify()

    if auto_optimize:
        pass

    tsdfg.validate()


__all__ = [
    "jace_auto_optimize",
]
