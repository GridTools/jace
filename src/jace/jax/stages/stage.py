# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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
