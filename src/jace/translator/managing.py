# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Module for managing the individual sutranslators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, overload

from jace import translator


# List of all primitive translators that are known to Jace.
#  They are filled through the `add_subtranslator()` decorator.
#  See also the note in `get_subtranslators_cls()`
_KNOWN_SUBTRANSLATORS: list[type[translator.PrimitiveTranslator]] = []


@overload
def add_subtranslator(
    subtrans: Literal[None], /, overwrite: bool = False
) -> Callable[[type[translator.PrimitiveTranslator]], type[translator.PrimitiveTranslator]]: ...


@overload
def add_subtranslator(
    subtrans: type[translator.PrimitiveTranslator], /, overwrite: bool = False
) -> type[translator.PrimitiveTranslator]: ...


def add_subtranslator(
    subtrans: type[translator.PrimitiveTranslator] | None = None,
    /,
    overwrite: bool = False,
) -> (
    type[translator.PrimitiveTranslator]
    | Callable[[type[translator.PrimitiveTranslator]], type[translator.PrimitiveTranslator]]
):
    """Decorator to add `subtrans` to the list of known subtranslators.

    If a class is tried to be registered twice an error will be generated unless, `overwrite` is set.
    """
    if subtrans is None:

        def wrapper(
            real_subtrans: type[translator.PrimitiveTranslator],
        ) -> type[translator.PrimitiveTranslator]:
            return add_subtranslator(real_subtrans, overwrite=overwrite)

        return wrapper

    if subtrans in _KNOWN_SUBTRANSLATORS:
        if overwrite:
            _KNOWN_SUBTRANSLATORS.remove(subtrans)
        else:
            raise ValueError(
                f"Tried to add '{type(subtrans).__name__}' twice to the list of known primitive translators."
            )

    _KNOWN_SUBTRANSLATORS.append(subtrans)
    return subtrans


def get_subtranslators_cls() -> Sequence[type[translator.PrimitiveTranslator]]:
    """Returns the list of all subtranslator known to JaCe.

    The subtranslators are returned in FIFO order.
    """
    # There is a chicken-egg problem, i.e. circular import, if we use the decorator to add the build in classes.
    #  The problem is, that they are only run, i.e. added to the list, upon importing.
    #  Thus we have to explicitly import the subtranslator, but this would then lead to a circular import.
    #  For that reason we import the subpackage here explicitly.
    #  However, this requires that all are imported by the `__init__.py` file.
    #  I do not know a way to do this better.
    #  Actually I want to do it somehow upon the importation of `jace` itself.
    from jace.translator import primitive_translators  # noqa: F401  # Unused import

    return list(reversed(_KNOWN_SUBTRANSLATORS))
