# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""General configuration for the tests of the primitive translators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jace import optimization, stages


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(
    autouse=True,
    params=[
        optimization.NO_OPTIMIZATIONS,
        optimization.DEFAULT_OPTIMIZATIONS,
    ],
)
def _set_compile_options(request) -> Generator[None, None, None]:
    """Set the options used for testing the primitive translators.

    This fixture override the global defined fixture.

    Todo:
        Implement a system that only runs the optimization case in CI.
    """
    initial_compile_options = stages.update_active_compiler_options(request.param)
    yield
    stages.update_active_compiler_options(initial_compile_options)
