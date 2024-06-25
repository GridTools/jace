# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
JaCe specific optimizations.

Todo:
    Organize this module once it is a package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict

from typing_extensions import Unpack


if TYPE_CHECKING:
    from jace import translated_jaxpr_sdfg as tjsdfg


DEFAULT_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": True,
    "simplify": True,
    "persistent_transients": True,
}

NO_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": False,
    "simplify": False,
    "persistent_transients": False,
}


class CompilerOptions(TypedDict, total=False):
    """
    All known compiler options to `JaCeLowered.compile()`.

    See `jace_optimize()` for a description of the different options.

    There are some predefined option sets in `jace.jax.stages`:
    - `DEFAULT_OPTIONS`
    - `NO_OPTIMIZATIONS`
    """

    auto_optimize: bool
    simplify: bool
    persistent_transients: bool


def jace_optimize(tsdfg: tjsdfg.TranslatedJaxprSDFG, **kwargs: Unpack[CompilerOptions]) -> None:  # noqa: D417 [undocumented-param]
    """
    Performs optimization of the translated SDFG _in place_.

    It is recommended to use the `CompilerOptions` `TypedDict` to pass options
    to the function. However, any option that is not specified will be
    interpreted as to be disabled.

    Args:
        tsdfg: The translated SDFG that should be optimized.
        simplify: Run the simplification pipeline.
        auto_optimize: Run the auto optimization pipeline (currently does nothing)
        persistent_transients: Set the allocation lifetime of (non register) transients
            in the SDFG to `AllocationLifetime.Persistent`, i.e. keep them allocated
            between different invocations.
    """
    # TODO(phimuell): Implement the functionality.
    # Currently this function exists primarily for the sake of existing.

    simplify = kwargs.get("simplify", False)
    auto_optimize = kwargs.get("auto_optimize", False)

    if simplify:
        tsdfg.sdfg.simplify()

    if auto_optimize:
        pass

    tsdfg.validate()
