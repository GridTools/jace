# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module collecting all built-in subtranslators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from jace import translator as jtrans

from .alu_translator import ALUTranslator


# List of all subtranslators that ships with JaCe.
_BUILTIN_SUBTRANSLATORS: Final[list[type[jtrans.JaCeSubTranslatorInterface]]] = [
    ALUTranslator,
]

# All externally supplied subtranslator implementation.
#  It is a `dict` to do fast access and remember the order, value is always `None`.
#  The list is manipulated through `{add,rm}_subtranslator()`.
_EXTERNAL_SUBTRANSLATORS: dict[type[jtrans.JaCeSubTranslatorInterface], None] = {}


def add_subtranslator(
    subtrans: type[jtrans.JaCeSubTranslatorInterface],
) -> bool:
    """Add `subtrans` to the externally defined subtranslators.

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
    """Remove `subtrans` as externally defined subtranslators.

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
    """Returns the list of all subtranslator known to JaCe.

    Args:
        with_external:  Include the translators that were externally supplied.
        builtins:       Include the build in translators.

    Notes:
        If the externally defined subtranslators are requested they will be
            first and ordered as FILO order.
    """
    # It is important that the externally defined are ordered before the builtins
    #  and are ordered in FILO order, especuially if multiple subtranslator per
    #  primitive are registered. Because this way they are inserted first
    #  into the internal list of the driver, and furthermore since `sorted()`
    #  is stable they will tend to end up more to the front.
    ret: list[type[jtrans.JaCeSubTranslatorInterface]] = []
    if with_external:
        ret.extend(reversed(_EXTERNAL_SUBTRANSLATORS.keys()))
    if builtins:
        ret.extend(_BUILTIN_SUBTRANSLATORS)
    return ret


__all__ = [
    "add_subtranslator",
    "rm_subtranslator",
]