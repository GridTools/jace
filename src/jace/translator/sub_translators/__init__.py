# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module collecting all built-in subtranslators."""

from __future__ import annotations

from collections.abc import Sequence

from jace import translator as jtrans
from jace.translator.sub_translators.alu_translator import ALUTranslator


# List of all subtranslators that ships with JaCe.
_KNOWN_SUBTRANSLATORS: list[type[jtrans.PrimitiveTranslator]] = [
    ALUTranslator,
]


def add_subtranslator(
    subtrans: type[jtrans.PrimitiveTranslator],
) -> bool:
    """Add `subtrans` to the externally defined subtranslators.

    The function returns `True` if it was added and `False` is not.
    """
    # NOTE: Because `PrimitiveTranslator` has a property, it is not possible to use
    #         `issubclass()` here, to check if the interface is ready implemented.
    if subtrans in _KNOWN_SUBTRANSLATORS:
        # TODO: Consider moving `subtrans` to the front (last element).
        return False
    _KNOWN_SUBTRANSLATORS.append(subtrans)
    return True


def _get_subtranslators_cls() -> Sequence[type[jtrans.PrimitiveTranslator]]:
    """Returns the list of all subtranslator known to JaCe.

    The translators are returned in FIFO order.
    """
    return list(reversed(_KNOWN_SUBTRANSLATORS))


__all__ = [
    "ALUTranslator",
    "add_subtranslator",
]
