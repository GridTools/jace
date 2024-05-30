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

    There are some predefined option sets:
    - `DEFAULT_COMPILER_OPTIONS`
    - `NO_OPTIMIZATIONS`
    """

    auto_optimize: bool
    simplify: bool
    persistent: bool


DEFAULT_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": True,
    "simplify": True,
    "persistent": True,
}

NO_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": False,
    "simplify": False,
    "persistent": False,
}


def jace_optimize(
    tsdfg: translator.TranslatedJaxprSDFG,
    **kwargs: Unpack[CompilerOptions],
) -> None:
    """Performs optimization of the `fsdfg` _in place_.

    Currently this function only supports simplification.
    Its main job is to exists that we have something that we can call in the tool chain.

    Args:
        simplify:       Run the simplification pipeline.
        auto_optimize:  Run the auto optimization pipeline (currently does nothing)
        persistent:     Make the memory allocation persistent, i.e. allocate the transients only
                            once at the beginning and then reuse the memory across the lifetime of the SDFG.

    Note:
        By default all optimizations are disabled and this function acts as a noops.
    """
    if not tsdfg.is_finalized:
        raise ValueError("Can only optimize finalized SDFGs.")
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
