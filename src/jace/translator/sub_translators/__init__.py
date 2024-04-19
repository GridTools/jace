# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module collecting all built-in subtranslators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Type

import jace
from jace import translator as jtrans

from .alu_translator import ALUTranslator


# List of all subtranslators that ships with JaCe.
_BUILTIN_SUBTRANSLATORS: Final[list[type[jtrans.JaCeSubTranslatorInterface]]] = [
    ALUTranslator,
]

# List of the externally supplied subtranslator implementation.
#  It is a `dict` to do fast access and remember the order, value is always `None`.
#  The list is manipulated through `{add,rm}_subtranslator()`.
_EXTERNAL_SUBTRANSLATORS: dict[type[jtrans.JaCeSubTranslatorInterface], None] = {}


def add_subtranslator(
    subtrans: type[jtrans.JaCeSubTranslatorInterface],
) -> bool:
    """Add `subtrans` to the internal list of externally supplied subtranslators.

    The function returns `True` if it was added and `False` is not.
    """
    from inspect import isclass

    if subtrans in _EXTERNAL_SUBTRANSLATORS:
        return False
    if not isclass(subtrans):
        return False
    if not issubclass(subtrans, jtrans.JaCeSubTranslatorInterface):
        return False
    _EXTERNAL_SUBTRANSLATORS[subtrans] = None
    return True


def rm_subtranslator(
    subtrans: type[jtrans.JaCeSubTranslatorInterface],
    strict: bool = False,
) -> bool:
    """Removes subtranslator `subtrans` from the list of known subtranslators.

    If `subtrans` is not known no error is generated unless `strict` is set to `True`.
    """
    if subtrans not in _EXTERNAL_SUBTRANSLATORS:
        if strict:
            raise KeyError(f"Subtranslator '{type(subtrans)}' is not known.")
        return False
    del _EXTERNAL_SUBTRANSLATORS[subtrans]
    return True


def _get_subtranslators_cls(
    with_external: bool = True,
    builtins: bool = True,
) -> Sequence[type[jtrans.JaCeSubTranslatorInterface]]:
    """Returns a list of all subtranslator classes in JaCe.

    Args:
        with_external:  Include the translators that were externally supplied.
        builtins:       Include the build in translators.
    """
    ret: list[type[jtrans.JaCeSubTranslatorInterface]] = []
    if builtins:
        ret.extend(_BUILTIN_SUBTRANSLATORS)
    if with_external:
        ret.extend(_EXTERNAL_SUBTRANSLATORS.keys())
    return ret


__all__ = [
    "add_subtranslator",
    "rm_subtranslator",
]
