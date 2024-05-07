# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Interface of the Stages.

In `jace.jax.stages.__init__.py` this file must be imported first.
However, isort/ruff fail to do that and can not be convinced otherwise. 
For that reason this file was renamed to ensure that it comes at first.
"""

from __future__ import annotations

from jax._src import stages as jax_stages


class Stage(jax_stages.Stage):
    """A distinct step in the compilation chain, see module description for more.

    This class inherent from its Jax counterpart.
    The concrete steps are implemented in:
    - JaceWrapped
    - JaceLowered
    - JaceCompiled
    """
