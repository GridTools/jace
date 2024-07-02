# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JaCe: JAX jit using DaCe (Data Centric Parallel Programming)."""

from __future__ import annotations

import jace.translator.primitive_translators as _  # noqa: F401 [unused-import]  # Needed to populate the internal translator registry.

from .__about__ import __author__, __copyright__, __license__, __version__, __version_info__
from .api import grad, jacfwd, jacrev, jit


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
