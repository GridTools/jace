# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def ensure_iterability(
    x: Any,
    ign_str: bool = True,
) -> Iterable[Any]:
    """Ensures that `x` is iterable.

    By default strings are _not_ considered iterable.

    Args:
        x:          To test.
        ign_str:    Ignore that a string is iterabile.
    """
    if ign_str and isinstance(x, str):
        x = [x]
    elif isinstance(x, Iterable):
        pass
    return x


def is_jaceified(obj: Any) -> bool:
    """Tests if `obj` is decorated by JaCe.

    Similar to `jace.util.is_jaxified`, but for JaCe object.
    """
    from jace import jax as jjax, util as jutil

    if jutil.is_jaxified(obj):
        return False
    # Currently it is quite simple because we can just check if `obj`
    #  is derived from `jace.jax.JitWrapped`, might become harder in the future.
    return isinstance(obj, jjax.JitWrapped)
