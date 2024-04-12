# JaCe - JAX jit using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.metadata

import jace as m


def test_version():
    assert importlib.metadata.version("jace") == m.__version__
