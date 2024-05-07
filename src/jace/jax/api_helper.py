# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper function for the api."""

from __future__ import annotations

import functools as ft
from collections.abc import Callable
from typing import Any


def jax_wrapper(
    jax_fun: Callable,
    fun: Callable | None = None,
    /,
    **kwargs: Any,
) -> Callable:
    """Creates a wrapper function for"""

    # fmt: off
    if fun is None:
        def _inner_jax_wrapper(fun: Callable) -> Callable:
            return jax_wrapper(jax_fun, fun, **kwargs)
        return _inner_jax_wrapper
    # fmt: on

    ft.update_wrapper(
        wrapper=fun,
        wrapped=jax_fun,
        **kwargs,
    )
    return fun
