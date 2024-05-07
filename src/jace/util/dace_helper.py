# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements all utility functions that are related to DaCe.

Most of the functions defined here allow an unified access to DaCe's internals
in a consistent and stable way.
"""

from __future__ import annotations

# The compiled SDFG is not available in the dace namespace or anywhere else
#  Thus we import it here directly
from dace.codegen.compiled_sdfg import CompiledSDFG as CompiledSDFG
