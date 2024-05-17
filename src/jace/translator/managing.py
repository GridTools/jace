# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Module for managing the individual sutranslators.

The high level idea is that there is a "list" of instances of `PrimitiveTranslator`,
which is known as `_PRIMITIVE_TRANSLATORS_DICT`.
If not specified the content of this list is used to perform the translation.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Literal, cast, overload


if TYPE_CHECKING:
    from jace import translator

# These are the currently active primitive translators of JaCe.
_PRIMITIVE_TRANSLATORS_DICT: dict[str, translator.PrimitiveTranslatorCallable] = {}


@overload
def register_primitive_translator(
    prim_translator: Literal[None] = None,
    /,
    primitive: str | None = None,
    overwrite: bool = False,
) -> Callable[
    [translator.PrimitiveTranslator | translator.PrimitiveTranslatorCallable],
    translator.PrimitiveTranslator,
]: ...


@overload
def register_primitive_translator(
    prim_translator: translator.PrimitiveTranslator | translator.PrimitiveTranslatorCallable,
    *,
    primitive: str | None = None,
    overwrite: bool = False,
) -> translator.PrimitiveTranslator: ...


def register_primitive_translator(
    prim_translator: translator.PrimitiveTranslator
    | translator.PrimitiveTranslatorCallable
    | None = None,
    *,
    primitive: str | None = None,
    overwrite: bool = False,
) -> (
    translator.PrimitiveTranslator
    | Callable[
        [translator.PrimitiveTranslator | translator.PrimitiveTranslatorCallable],
        translator.PrimitiveTranslator,
    ]
):
    """Adds the primitive translator `prim_translator` to Jace's internal list of translators.

    If the primitive is already known an error is generated, if `overwrite` is set, it will be replaced.

    Args:
        prim_translator:    The primitive translator to annotate.
        primitive:          Name of the primitive `prim_translator` is handled.
                                If not given will use `prim_translator.primitive`.
        overwrite:          Replace the current primitive translator with `prim_translator`.

    Notes:
        Can only be used to register instances.
    """
    from jace import translator

    def wrapper(
        prim_translator: translator.PrimitiveTranslator | translator.PrimitiveTranslatorCallable,
    ) -> translator.PrimitiveTranslator:
        if not hasattr(prim_translator, "primitive"):
            if not primitive:
                raise ValueError(f"Missing primitive name for '{prim_translator}'")
            prim_translator.primitive = primitive  # type: ignore[attr-defined]
        elif prim_translator.primitive != (primitive or prim_translator.primitive):
            raise TypeError(
                f"Translator's primitive '{prim_translator.primitive}' doesn't match the supplied '{primitive}'."
            )

        if prim_translator.primitive in _PRIMITIVE_TRANSLATORS_DICT and not overwrite:
            raise ValueError(
                f"Explicit override=True needed for primitive '{prim_translator.primitive}' to overwrite existing one."
            )
        _PRIMITIVE_TRANSLATORS_DICT[prim_translator.primitive] = prim_translator

        # We add a `.primitive` property, thus it is for sure now no longer just a `PrimitiveTranslatorCallable`.
        return cast(translator.PrimitiveTranslator, prim_translator)

    return wrapper if prim_translator is None else wrapper(prim_translator)


def get_regsitered_primitive_translators() -> (
    MutableMapping[str, translator.PrimitiveTranslatorCallable]
):
    """Returns the currently active view of all _currently_ installed primitive translators in Jace.

    The returned mapping represents the active primitive translators at the time of calling.
    This means that calls to `register_primitive_translator()` does not modify the returned object.
    """
    return _PRIMITIVE_TRANSLATORS_DICT.copy()
