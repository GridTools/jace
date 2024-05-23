# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JaCe - JAX Just-In-Time compilation using DaCe."""

from __future__ import annotations

import jax as _jax

import jace.translator.primitive_translators as _  # noqa: F401  # Populate the internal registry.

from .__about__ import __author__, __copyright__, __license__, __version__, __version_info__
from .jax import grad, jacfwd, jacrev, jit


# In Jax `float32` is the main datatype, and they go to great lengths to avoid
#  some aggressive [type promotion](https://jax.readthedocs.io/en/latest/type_promotion.html).
#  However, in this case we will have problems when we call the SDFG, for some reasons
#  `CompiledSDFG` does not work in that case correctly, thus we enable it now globally.
_jax.config.update("jax_enable_x64", True)


__all__ = [
    "__author__",
    "__copyright__",
    "grad",
    "jit",
    "jacfwd",
    "jacrev",
    "__license__",
    "__version__",
    "__version_info__",
]

del _jax
