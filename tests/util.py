# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for the testing infrastructure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from jace import translator


__all__ = ["make_array"]


def make_array(
    shape: Sequence[int] | int,
    dtype: type = np.float64,
    order: str = "C",
    low: Any = None,
    high: Any = None,
) -> np.ndarray:
    """Generates a NumPy ndarray with shape `shape`.

    The function uses the generator that is managed by the `_reset_random_seed()`
    fixture. Thus inside a function the value will be deterministic.

    Args:
        shape: The shape to use.
        dtype: The data type to use.
        order: The order of the underlying array
        low: Minimal value.
        high: Maximal value.

    Note:
        The exact meaning of `low` and `high` depend on the type. For `bool` they
        are ignored. For float both must be specified and then values inside
        `[low, high)` are generated. For integer it is possible to specify only one.
        The appropriate numeric limit is used for the other.
    """

    if shape == ():
        return dtype(make_array((1,), dtype)[0])
    if isinstance(shape, int):
        shape = (shape,)

    if dtype == np.bool_:
        res = np.random.random(shape) > 0.5  # noqa: NPY002 [numpy-legacy-random]
    elif np.issubdtype(dtype, np.integer):
        iinfo: np.iinfo = np.iinfo(dtype)
        res = np.random.randint(  # noqa: NPY002 [numpy-legacy-random]
            low=iinfo.min if low is None else low,
            high=iinfo.max if high is None else high,
            size=shape,
            dtype=dtype,
        )
    elif np.issubdtype(dtype, np.complexfloating):
        res = make_array(shape, np.float64) + 1.0j * make_array(shape, np.float64)
    else:
        res = np.random.random(shape)  # type: ignore[assignment]  # noqa: NPY002 [numpy-legacy-random]
        if low is not None and high is not None:
            res = low + (high - low) * res
        assert (low is None) == (high is None)

    return np.array(res, order=order, dtype=dtype)  # type: ignore[call-overload]  # Because we use `str` as `order`.


def set_active_primitive_translators_to(
    new_active_primitives: Mapping[str, translator.PrimitiveTranslator],
) -> dict[str, translator.PrimitiveTranslator]:
    """Exchanges the currently active set of translators with `new_active_primitives`.

    The function will return the set of translators the were active before the call.

    Args:
        new_active_primitives: The new set of active translators.
    """
    assert all(
        primitive_name == translator.primitive
        for primitive_name, translator in new_active_primitives.items()
    )
    previously_active_translators = translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY
    translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY = {**new_active_primitives}
    return previously_active_translators
