# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""General configuration for the tests.

Todo:
    - Implement some fixture that allows to force validation.
"""

from __future__ import annotations

from collections.abc import Generator

import jax
import numpy as np
import pytest

from jace import optimization, stages
from jace.util import translation_cache as tcache


@pytest.fixture(autouse=True)
def _enable_x64_mode_in_jax() -> Generator[None, None, None]:
    """Fixture of enable the `x64` mode in JAX.

    Currently, JaCe requires that `x64` mode is enabled and will do all JAX
    things with it enabled. However, if we use JAX with the intend to compare
    it against JaCe we must also enable it for JAX.
    """
    with jax.experimental.enable_x64():
        yield


@pytest.fixture(autouse=True)
def _disable_jit() -> Generator[None, None, None]:
    """Fixture for disable the dynamic jiting in JAX, used by default.

    Using this fixture has two effects.
    - JAX will not cache the results, i.e. every call to a jitted function will
        result in a tracing operation.
    - JAX will not use implicit jit operations, i.e. nested Jaxpr expressions
        using `pjit` are avoided.

    This essentially disable the `jax.jit` decorator, however, the `jace.jit`
    decorator is still working.

    Note:
        The second point, i.e. preventing JAX from running certain things in `pjit`,
        is the main reason why this fixture is used by default, without it
        literal substitution is useless and essentially untestable.
        In certain situation it can be disabled.
    """
    with jax.disable_jit(disable=True):
        yield


@pytest.fixture()
def _enable_jit() -> Generator[None, None, None]:
    """Fixture to enable jit compilation.

    Essentially it undoes the effects of the `_disable_jit()` fixture.
    It is important that this fixture is not automatically activated.
    """
    with jax.disable_jit(disable=False):
        yield


@pytest.fixture(autouse=True)
def _clear_translation_cache() -> Generator[None, None, None]:
    """Decorator that clears the translation cache.

    Ensures that a function finds an empty cache and clears up afterwards.
    """
    tcache.clear_translation_cache()
    yield
    tcache.clear_translation_cache()


@pytest.fixture(autouse=True)
def _reset_random_seed() -> None:
    """Fixture for resetting the random seed.

    This ensures that for every test the random seed of NumPy is reset.
    This seed is used by the `util.mkarray()` helper.
    """
    np.random.seed(42)  # noqa: NPY002 [numpy-legacy-random]


@pytest.fixture(autouse=True)
def _set_compile_options() -> Generator[None, None, None]:
    """Disable all optimizations of jitted code.

    Without explicitly supplied arguments `JaCeLowered.compile()` will not
    perform any optimizations.
    Please not that certain tests might override this fixture.
    """
    initial_compile_options = stages.update_active_compiler_options(optimization.NO_OPTIMIZATIONS)
    yield
    stages.update_active_compiler_options(initial_compile_options)
