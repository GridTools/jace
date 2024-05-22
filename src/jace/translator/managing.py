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

from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Literal, cast, overload


if TYPE_CHECKING:
    from jace import translator

# These are the currently active primitive translators of JaCe.
_PRIMITIVE_TRANSLATORS_DICT: dict[str, translator.PrimitiveTranslator] = {}


@overload
def make_primitive_translator(
    primitive: str,
    prim_translator: Literal[None] = None,
) -> Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]: ...


@overload
def make_primitive_translator(
    primitive: str, prim_translator: translator.PrimitiveTranslatorCallable
) -> translator.PrimitiveTranslator: ...


def make_primitive_translator(
    primitive: str,
    prim_translator: translator.PrimitiveTranslatorCallable | None = None,
) -> (
    Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]
    | translator.PrimitiveTranslator
):
    """Decorator to turn a Callable into a `PrimitiveTranslator` for primitive `primitive`.

    This function can be used to decorate functions that should serve as primitive translators.
    Essentially, the decorator adds a `primitive` property to the decorated function and returns it.
    However, this function does not register the primitive into the global registry,
    for this you have to use `register_primitive_translator()`.
    """

    def wrapper(
        prim_translator: translator.PrimitiveTranslatorCallable,
    ) -> translator.PrimitiveTranslator:
        if getattr(prim_translator, "primitive", primitive) != primitive:
            raise ValueError(
                f"Tried to change the 'primitive' property of '{prim_translator}' from '{prim_translator.primitive}' to '{primitive}'."  # type: ignore[attr-defined]
            )
        prim_translator.primitive = primitive  # type: ignore[attr-defined]  # we add the attribute, so it is not defined yet.
        return cast(translator.PrimitiveTranslator, prim_translator)

    return wrapper if prim_translator is None else wrapper(prim_translator)


def register_primitive_translator(
    prim_translator: translator.PrimitiveTranslator,
    overwrite: bool = False,
) -> translator.PrimitiveTranslator:
    """Adds the primitive translator to Jace's internal list of translators and return it again.

    If the primitive is already known an error is generated, if `overwrite` is set, it will be replaced.
    To add a `primitive` property use the `@make_primitive_translator` decorator.

    Args:
        prim_translator:    The primitive translator to annotate.
        overwrite:          Replace the current primitive translator with `prim_translator`.
    """

    def wrapper(
        prim_translator: translator.PrimitiveTranslator,
    ) -> translator.PrimitiveTranslator:
        if prim_translator.primitive in _PRIMITIVE_TRANSLATORS_DICT and not overwrite:
            raise ValueError(
                f"Explicit override=True needed for primitive '{prim_translator.primitive}' to overwrite existing one."
            )
        _PRIMITIVE_TRANSLATORS_DICT[prim_translator.primitive] = prim_translator
        return prim_translator

    return wrapper if prim_translator is None else wrapper(prim_translator)


def get_regsitered_primitive_translators() -> dict[str, translator.PrimitiveTranslator]:
    """Returns a view of the _currently_ active set of installed primitive translators in Jace.

    The returned mapping represents the active primitive translators at the time of calling.
    This means that calls to `register_primitive_translator()` or any other mutating call will not affect the returned object.
    """
    return _PRIMITIVE_TRANSLATORS_DICT.copy()


def set_active_primitive_translators_to(
    new_translators: Mapping[str, translator.PrimitiveTranslator],
) -> MutableMapping[str, translator.PrimitiveTranslator]:
    """Exchange the currently active subtranslators in Jace with `new_translators` and returns the previous ones.

    This function allows you to restore a specific state that was obtained by a previous call to `get_regsitered_primitive_translators()`.
    While the function returns a mutable object, any changes to the returned object have no effect on the global state of the registry.
    """
    global _PRIMITIVE_TRANSLATORS_DICT
    assert all(getattr(trans, "primitive", prim) for prim, trans in new_translators.items())
    previous_translators = _PRIMITIVE_TRANSLATORS_DICT
    _PRIMITIVE_TRANSLATORS_DICT = dict(new_translators)
    return previous_translators
