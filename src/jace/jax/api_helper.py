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
    jace_fun: Callable | None = None,
    /,
    rewriting: bool = True,
    **kwargs: Any,
) -> Callable:
    """Creates a wrapper to encapsulate Jax in Jace functions.

    A replacement for `functools.wraps` but for the special
    case that a Jace function should replace a Jax function.

    Args:
        rewriting:      Replace 'JAX' with 'JaCe' in the doc string.

    Todo:
        Improve.
    """

    # fmt: off
    if jace_fun is None:
        def _inner_jax_wrapper(jace_fun_: Callable) -> Callable:
            return jax_wrapper(jax_fun, jace_fun_, **kwargs)
        return _inner_jax_wrapper
    # fmt: on

    # This function creates the `__wrapped__` property, that I do not want
    #  So we have to replace it, I think we should consider using the one of Jax.
    ft.update_wrapper(
        wrapper=jace_fun,
        wrapped=jax_fun,
        **kwargs,
    )

    if rewriting:
        # TODO(phimuell): Handle web addresses, code example and more.
        jace_fun.__doc__ = jace_fun.__doc__.replace("JAX", "JaCe")

    return jace_fun
