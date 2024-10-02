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

import dace
from dace.transformation.auto import auto_optimize as dace_autoopt
from typing_extensions import Unpack


if TYPE_CHECKING:
    from jace import translated_jaxpr_sdfg as tjsdfg


DEFAULT_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": False,
    "simplify": True,
    "persistent_transients": True,
    "validate": True,
    "validate_all": False,
}

NO_OPTIMIZATIONS: Final[CompilerOptions] = {
    "auto_optimize": False,
    "simplify": False,
    "persistent_transients": False,
    "validate": True,
    "validate_all": False,
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
    validate: bool
    validate_all: bool


def jace_optimize(  # noqa: D417 [undocumented-param]  # `kwargs` is not documented.
    tsdfg: tjsdfg.TranslatedJaxprSDFG,
    device: dace.DeviceType,
    **kwargs: Unpack[CompilerOptions],
) -> None:  # [undocumented-param]
    """
    Performs optimization of the translated SDFG _in place_.

    It is recommended to use the `CompilerOptions` `TypedDict` to pass options
    to the function. However, any option that is not specified will be
    interpreted as to be disabled.

    Args:
        tsdfg: The translated SDFG that should be optimized.
        device: The device on which the SDFG will run on.
        simplify: Run the simplification pipeline.
        auto_optimize: Run the auto optimization pipeline.
        persistent_transients: Set the allocation lifetime of (non register) transients
            in the SDFG to `AllocationLifetime.Persistent`, i.e. keep them allocated
            between different invocations.
        validate: Perform validation at the end.
        validate_all: Perform extensive validation.

    Note:
        Currently DaCe's auto optimization pipeline is used when auto optimize is
        enabled. However, it might change in the future. Because DaCe's auto
        optimizer is considered unstable it must be explicitly enabled.
    """
    assert device in {dace.DeviceType.CPU, dace.DeviceType.GPU}
    simplify = kwargs.get("simplify", False)
    auto_optimize = kwargs.get("auto_optimize", False)
    validate = kwargs.get("validate", DEFAULT_OPTIMIZATIONS["validate"])
    validate_all = kwargs.get("validate_all", DEFAULT_OPTIMIZATIONS["validate_all"])

    if simplify:
        tsdfg.sdfg.simplify(
            validate=validate,
            validate_all=validate_all,
        )

    if device == dace.DeviceType.GPU:
        tsdfg.sdfg.apply_gpu_transformations(
            validate=validate,
            validate_all=validate_all,
            simplify=True,
        )

    if auto_optimize:
        dace_autoopt.auto_optimize(
            sdfg=tsdfg.sdfg,
            device=device,
            validate=validate,
            validate_all=validate_all,
        )

    if validate or validate_all:
        tsdfg.validate()
