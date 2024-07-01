# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""General configuration for the tests of the primitive translators."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from jace import optimization, stages


@pytest.fixture(
    autouse=True,
    params=[
        optimization.NO_OPTIMIZATIONS,
        pytest.param(
            optimization.DEFAULT_OPTIMIZATIONS,
            marks=pytest.mark.skip("Simplify bug 'https://github.com/spcl/dace/issues/1595'"),
        ),
    ],
)
def _set_compile_options(request) -> Generator[None, None, None]:
    """Set the options used for testing the primitive translators.

    This fixture override the global defined fixture.

    Todo:
        Implement a system that only runs the optimization case in CI.
    """
    with stages.temporary_compiler_options(request.param):
        yield
