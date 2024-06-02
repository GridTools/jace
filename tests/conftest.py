# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""General configuration for the tests.

Todo:
    - Implement some fixture that allows to force validation.
    - Implement fixture to disable and enable optimisation, i.e. doing it twice.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jace.util import translation_cache as tcache


@pytest.fixture(autouse=True)
def _enable_x64_mode_in_jax():
    """Fixture of enable the `x64` mode in Jax.

    Currently, JaCe requires that `x64` mode is enabled and will do all Jax things with it enabled.
    However, if we use Jax with the intend to compare it against JaCe we must also enable it for
    Jax.
    """
    with jax.experimental.enable_x64():
        yield


@pytest.fixture(autouse=True)
def _disable_jit():
    """Fixture for disable the dynamic jiting in Jax.

    For certain reasons Jax puts certain primitives inside a `pjit` primitive, i.e. nested Jaxpr.
    The intent is, that these operations can/should run on an accelerator.

    But this is a problem, since JaCe can not handle this primitive, it leads to an error.
    To overcome this problem, we will globally disable this feature until we can handle `pjit`.

    Todo:
        Remove as soon as we can handle nested `jit`.
    """
    with jax.disable_jit(disable=True):
        yield


@pytest.fixture(autouse=True)
def _clear_translation_cache():
    """Decorator that clears the translation cache.

    Ensures that a function finds an empty cache and clears up afterwards.
    """
    tcache.clear_translation_cache()
    yield
    tcache.clear_translation_cache()


@pytest.fixture(autouse=True)
def _reset_random_seed():
    """Fixture for resetting the random seed.

    This ensures that for every test the random seed of NumPy is reset.
    This seed is used by the `util.mkarray()` helper.
    """
    np.random.seed(42)  # noqa: NPY002  # We use this seed for the time being.
