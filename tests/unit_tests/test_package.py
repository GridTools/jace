# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.metadata

import pytest

import jace as m


@pytest.mark.skip(reason="This does not work yet.")
def test_version() -> None:
    assert importlib.metadata.version("jace") == m.__version__
