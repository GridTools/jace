# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JaCe: JAX jit using DaCe (Data Centric Parallel Programming)."""

from __future__ import annotations

import jax

import jace.translator.primitive_translators as _  # noqa: F401 [unused-import]  # Needed to populate the internal translator registry.

from .__about__ import __author__, __copyright__, __license__, __version__, __version_info__
from .api import grad, jacfwd, jacrev, jit


if jax.version._version_as_tuple(jax.__version__) < (0, 4, 33):
    raise ImportError(f"Require at least JAX version '0.4.33', but found '{jax.__version__}'.")


__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__version_info__",
    "grad",
    "jacfwd",
    "jacrev",
    "jit",
]
