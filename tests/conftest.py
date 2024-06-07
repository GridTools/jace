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

from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

from jace import optimization, stages
from jace.util import translation_cache as tcache


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _enable_x64_mode_in_jax() -> Generator[None, None, None]:
    """Fixture of enable the `x64` mode in Jax.

    Currently, JaCe requires that `x64` mode is enabled and will do all Jax
    things with it enabled. However, if we use Jax with the intend to compare
    it against JaCe we must also enable it for Jax.
    """
    with jax.experimental.enable_x64():
        yield


@pytest.fixture(autouse=True)
def _disable_jit() -> Generator[None, None, None]:
    """Fixture for disable the dynamic jiting in Jax.

    For certain reasons Jax puts certain primitives inside a `pjit` primitive,
    i.e. nested Jaxpr. The intent is, that these operations can/should run on
    an accelerator.

    But this is a problem, since JaCe can not handle this primitive, it leads
    to an error. To overcome this problem, we will globally disable this feature
    until we can handle `pjit`.

    Note this essentially disable the `jax.jit` decorator, however, the `jace.jit`
    decorator is still working.

    Todo:
        Remove as soon as we can handle nested `jit`.
    """
    with jax.disable_jit(disable=True):
        yield


@pytest.fixture()
def _enable_jit() -> Generator[None, None, None]:
    """Fixture to enable jit compilation.

    Essentially it undoes the effects of the `_disable_jit()` fixture.
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
    np.random.seed(42)  # noqa: NPY002  # We use this seed for the time being.


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
