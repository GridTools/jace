# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for the testing infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jace import translator


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


__all__ = ["mkarray"]


def mkarray(shape: Sequence[int] | int, dtype: type = np.float64, order: str = "C") -> np.ndarray:
    """Generates a NumPy ndarray with shape `shape`.

    The function uses the generator that is managed by the `_reset_random_seed()`
    fixture. Thus inside a function the value will be deterministic.

    Args:
        shape:      The shape to use.
        dtype:      The data type to use.

    Notes:
        Floating point based values are generated in the range 0 to 1.0.
    """

    if shape == ():
        return mkarray((1,), dtype)[0]
    if isinstance(shape, int):
        shape = (shape,)

    if dtype == np.bool_:
        res = np.random.random(shape) > 0.5  # noqa: NPY002
    elif np.issubdtype(dtype, np.integer):
        iinfo: np.iinfo = np.iinfo(dtype)
        res = np.random.randint(  # noqa: NPY002
            low=iinfo.min, high=iinfo.max, size=shape, dtype=dtype
        )
    elif np.issubdtype(dtype, np.complexfloating):
        res = mkarray(shape, np.float64) + 1.0j * mkarray(shape, np.float64)
    else:
        res = np.random.random(shape)  # type: ignore[assignment]  # noqa: NPY002
    return np.array(res, order=order, dtype=dtype)  # type: ignore[call-overload]


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
