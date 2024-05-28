# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Module for managing the global primitive translators.

The high level idea is that there is a registry of all currently active primitive translators.
If `primitive_translators` is not given to `jit` it will use this global registry.
A primitive, i.e. an object that satisfies the `PrimitiveTranslator` interface, can be added
to the registry by `register_primitive_translator()`. To retrieve the translators that are
currently active you can use the `get_regsitered_primitive_translators()` function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Literal, cast, overload


if TYPE_CHECKING:
    from jace import translator

#: Global registry of the active primitive translators.
#:  The `dict` maps the name of a primitive to its associated translators.
_PRIMITIVE_TRANSLATORS_DICT: dict[str, translator.PrimitiveTranslator] = {}


@overload
def make_primitive_translator(
    primitive: str,
    primitive_translator: Literal[None] = None,
) -> Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]: ...


@overload
def make_primitive_translator(
    primitive: str, primitive_translator: translator.PrimitiveTranslatorCallable
) -> translator.PrimitiveTranslator: ...


def make_primitive_translator(
    primitive: str,
    primitive_translator: translator.PrimitiveTranslatorCallable | None = None,
) -> (
    Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]
    | translator.PrimitiveTranslator
):
    """Turn `primitive_translator` into a `PrimitiveTranslator` for primitive `primitive`.

    Essentially, this function adds the `primitive` property to a callable, such that it satisfy
    the `PrimitiveTranslator` protocol. However, it does not add it to the registry, for that
    `register_primitive_translator()` has to be used.

    Notes:
        This function cal also be used as decorator.
    """

    def wrapper(
        primitive_translator: translator.PrimitiveTranslatorCallable,
    ) -> translator.PrimitiveTranslator:
        from jace import translator  # Cyclic

        if getattr(primitive_translator, "primitive", primitive) != primitive:
            raise ValueError(
                f"Tried to change the 'primitive' property of '{primitive_translator}' from "
                f"'{primitive_translator.primitive}' to '{primitive}'."  # type: ignore[attr-defined]
            )
        primitive_translator.primitive = primitive  # type: ignore[attr-defined]  # We define the attribute.
        return cast(translator.PrimitiveTranslator, primitive_translator)

    return wrapper if primitive_translator is None else wrapper(primitive_translator)


@overload
def register_primitive_translator(
    primitive_translator: Literal[None] = None,
    overwrite: bool = False,
) -> Callable[[translator.PrimitiveTranslator], translator.PrimitiveTranslator]: ...


@overload
def register_primitive_translator(
    primitive_translator: translator.PrimitiveTranslator,
    overwrite: bool = False,
) -> translator.PrimitiveTranslator: ...


def register_primitive_translator(
    primitive_translator: translator.PrimitiveTranslator | None = None,
    overwrite: bool = False,
) -> (
    translator.PrimitiveTranslator
    | Callable[[translator.PrimitiveTranslator], translator.PrimitiveTranslator]
):
    """Adds a primitive translator to Jace's global registry.

    If a translator for `primitive` is already registered an error will be generated. However,
    by specifying `overwrite` `primitive_translator` will replace the current one.

    Args:
        primitive_translator: The primitive translator to add to the global registry.
        overwrite:            Replace the current primitive translator with `primitive_translator`.

    Note:
        To add a `primitive` property use the `@make_primitive_translator` decorator.
        This function returns `primitive_translator` unmodified, which allows it to be
        used as decorator.
    """

    def wrapper(
        primitive_translator: translator.PrimitiveTranslator,
    ) -> translator.PrimitiveTranslator:
        if primitive_translator.primitive in _PRIMITIVE_TRANSLATORS_DICT and not overwrite:
            raise ValueError(
                f"Explicit override=True needed for primitive '{primitive_translator.primitive}' "
                "to overwrite existing one."
            )
        _PRIMITIVE_TRANSLATORS_DICT[primitive_translator.primitive] = primitive_translator
        return primitive_translator

    return wrapper if primitive_translator is None else wrapper(primitive_translator)


def get_regsitered_primitive_translators() -> dict[str, translator.PrimitiveTranslator]:
    """Returns a copy of the current state of Jace's global primitive registry.

    The function returns a mapping that maps the name of a primitive to the associated translator.
    No change to the global registry will affect the return value and vice versa.
    """
    return _PRIMITIVE_TRANSLATORS_DICT.copy()


def set_active_primitive_translators_to(
    new_translators: Mapping[str, translator.PrimitiveTranslator],
) -> MutableMapping[str, translator.PrimitiveTranslator]:
    """Exchange the global translator registry of Jace with `new_translators`.

    The function will return the state of the global translator registry just before this call.
    Any changes to `new_translators` after calling this function will have no effect on the
    global translator registry and vice versa.
    """
    global _PRIMITIVE_TRANSLATORS_DICT
    assert all(getattr(trans, "primitive", prim) for prim, trans in new_translators.items())
    previous_translators = _PRIMITIVE_TRANSLATORS_DICT
    _PRIMITIVE_TRANSLATORS_DICT = dict(new_translators)
    return previous_translators
