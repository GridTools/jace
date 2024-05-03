# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing all regex pattern that we need inside JaCe."""

from __future__ import annotations

import re


# Valid name for a jax variable.
_VALID_JAX_VAR_NAME: re.Pattern = re.compile("(jax[0-9]+_?)|([a-z]+_?)")

# Valid name for an SDFG variable.
_VALID_SDFG_VAR_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

# Valid name for an SDFG itself, includes `SDFGState` objects.
_VALID_SDFG_OBJ_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")
