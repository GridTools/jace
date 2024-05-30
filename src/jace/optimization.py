# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module that will host all optimization functions specific to JaCe.

Currently just a dummy existing for the sake of providing some callable function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict

from typing_extensions import Unpack


if TYPE_CHECKING:
    from jace import translator


class CompilerOptions(TypedDict, total=False):
    """All known compiler options known to `JaCeLowered.compile()`.

    See `jace_optimize()` for a description of the different options.

    There are some predefined option sets in `jace.jax.stages`:
    - `DEFAULT_COMPILER_OPTIONS`
    - `NO_OPTIMIZATIONS`
    """

    auto_optimize: bool
    simplify: bool


DEFAULT_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": True,
    "simplify": True,
}

NO_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": False,
    "simplify": False,
}


def jace_optimize(
    tsdfg: translator.TranslatedJaxprSDFG,
    **kwargs: Unpack[CompilerOptions],
) -> None:
    """Performs optimization of the `tsdfg` _in place_.

    Currently this function only supports simplification.
    Its main job is to exists that we have something that we can call in the tool chain.

    Args:
        tsdfg:          The translated SDFG that should be optimized.
        simplify:       Run the simplification pipeline.
        auto_optimize:  Run the auto optimization pipeline (currently does nothing)

    Note:
        By default all optimizations are disabled and this function acts as a noops.
    """
    if not kwargs:
        return

    # Unpack the arguments, defaults are such that no optimization is done.
    simplify = kwargs.get("simplify", False)
    auto_optimize = kwargs.get("auto_optimize", False)

    if simplify:
        tsdfg.sdfg.simplify()

    if auto_optimize:
        pass

    tsdfg.validate()
