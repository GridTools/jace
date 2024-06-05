# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JaCe specific optimizations.

Currently just a dummy exists for the sake of providing a callable function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict

from typing_extensions import Unpack


if TYPE_CHECKING:
    from jace import translator


class CompilerOptions(TypedDict, total=False):
    """All known compiler options to `JaCeLowered.compile()`.

    See `jace_optimize()` for a description of the different options.

    There are some predefined option sets in `jace.jax.stages`:
    - `DEFAULT_OPTIONS`
    - `NO_OPTIMIZATIONS`
    """

    auto_optimize: bool
    simplify: bool
    persistent: bool


# TODO(phimuell): Add a context manager to modify the default.
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
    """Performs optimization of the translated SDFG _in place_.

    It is recommended to use the `CompilerOptions` `TypedDict` to pass options
    to the function. However, any option that is not specified will be
    interpreted as to be disabled.

    Args:
        tsdfg: The translated SDFG that should be optimized.
        simplify: Run the simplification pipeline.
        auto_optimize: Run the auto optimization pipeline (currently does nothing)
        persistent:  Make the memory allocation persistent, i.e. allocate the transients only
            once at the beginning and then reuse the memory across the lifetime of the SDFG.
    """
    # Currently this function exists primarily for the same of existing.

    simplify = kwargs.get("simplify", False)
    auto_optimize = kwargs.get("auto_optimize", False)

    if simplify:
        tsdfg.sdfg.simplify()

    if auto_optimize:
        pass

    tsdfg.validate()
