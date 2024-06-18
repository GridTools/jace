# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Interface for all primitive translators and managing of the global translator registry.

Todo:
    Implement proper context manager for working with the registry.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import dace
    from jax import core as jax_core

    from jace import translator

#: Global registry of the active primitive translators.
#:  The `dict` maps the name of a primitive to its associated translators.
_PRIMITIVE_TRANSLATORS_REGISTRY: dict[str, translator.PrimitiveTranslator] = {}


class PrimitiveTranslatorCallable(Protocol):
    """Callable version of the primitive translators."""

    @abc.abstractmethod
    def __call__(
        self,
        builder: translator.JaxprTranslationBuilder,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        """
        Translates the Jax primitive into its SDFG equivalent.

        Before the builder calls this function it will perform the following
        preparatory tasks:
        - It will allocate the SDFG variables that are used as outputs. Their
            names will be passed through the `out_var_names` argument, in the
            same order as `eqn.outvars`.
        - It will collect the names of the SDFG variables that are used as
            inputs and place them in `in_var_names`, in the same order as
            `eqn.invars`. If an input argument refers to a literal no SDFG
            variable is created for it and `None` is used to indicate this.
        - The builder will create variables that are used as output. They are
            passed as `out_var_names`, same order as in the equation.
        - The builder will create a new terminal state and pass it as `eqn_state`
            argument. This state is guaranteed to be empty and
            `translator.terminal_sdfg_state is eqn_state` holds.

        Then the primitive translator is called.
        Usually a primitive translator should construct the dataflow graph
        inside `eqn_state`. However, it is allowed that the primitive translators
        creates more states if needed, but this state machinery has to have a
        single terminal state, which must be returned and reachable from
        `eqn_state`. If the function returns `None` the builder will assume that
        primitive translator was able to fully construct the dataflow graph
        within `eqn_state`.

        A primitive translator has to use the passed input variables,
        `in_var_names` and must write its output into the variables indicated
        by `out_var_names`. But it is allowed that a primitive translator
        creates intermediate values as needed. To ensure that there are no
        collision with further variables, the translator should prefix them,
        see the `name_prefix` argument of `JaxprTranslationBuilder.add_array()`.

        Args:
            builder: The builder object of the translation.
            in_var_names: List of the names of the arrays created inside the
                SDFG for the inpts or `None` in case of a literal.
            out_var_names: List of the names of the arrays created inside the
                SDFG for the outputs.
            eqn: The Jax primitive that should be translated.
            eqn_state: State into which the primitive`s SDFG representation
                should be constructed.
        """
        ...


@runtime_checkable
class PrimitiveTranslator(PrimitiveTranslatorCallable, Protocol):
    """
    Interface for all Jax primitive translators.

    A translator for a primitive translates a single equation of a Jaxpr into
    its SDFG equivalent. For satisfying this interface a concrete implementation
    must be immutable after construction.

    Primitive translators are simple, but highly specialized objects that are
    only able to perform the translation of a single primitive. The overall
    translation process itself is managed by a builder object, which also owns
    and manage the primitive translators. In the end this implements the
    delegation pattern.

    The `jace.translator.register_primitive_translator()` function can be used
    to add a translator to the JaCe global registry.
    """

    @property
    @abc.abstractmethod
    def primitive(self) -> str:
        """Returns the name of the Jax primitive that `self` is able to handle."""
        ...


@overload
def make_primitive_translator(
    primitive: str, primitive_translator: Literal[None] = None
) -> Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]: ...


@overload
def make_primitive_translator(
    primitive: str, primitive_translator: translator.PrimitiveTranslatorCallable
) -> translator.PrimitiveTranslator: ...


def make_primitive_translator(
    primitive: str, primitive_translator: translator.PrimitiveTranslatorCallable | None = None
) -> (
    Callable[[translator.PrimitiveTranslatorCallable], translator.PrimitiveTranslator]
    | translator.PrimitiveTranslator
):
    """
    Turn `primitive_translator` into a `PrimitiveTranslator` for primitive `primitive`.

    Essentially, this function adds the `primitive` property to a callable, such
    that it satisfy the `PrimitiveTranslator` protocol. However, it does not add
    it to the registry, for that `register_primitive_translator()` has to be used.

    Notes:
        This function can also be used as decorator.
    """

    def wrapper(
        primitive_translator: translator.PrimitiveTranslatorCallable,
    ) -> translator.PrimitiveTranslator:
        if getattr(primitive_translator, "primitive", primitive) != primitive:
            raise ValueError(
                f"Tried to change the 'primitive' property of '{primitive_translator}' from "
                f"'{primitive_translator.primitive}' to '{primitive}'."  # type: ignore[attr-defined]
            )
        primitive_translator.primitive = primitive  # type: ignore[attr-defined]  # We define the attribute.
        return cast("translator.PrimitiveTranslator", primitive_translator)

    return wrapper if primitive_translator is None else wrapper(primitive_translator)


@overload
def register_primitive_translator(
    primitive_translator: Literal[None] = None, overwrite: bool = False
) -> Callable[[translator.PrimitiveTranslator], translator.PrimitiveTranslator]: ...


@overload
def register_primitive_translator(
    primitive_translator: translator.PrimitiveTranslator, overwrite: bool = False
) -> translator.PrimitiveTranslator: ...


def register_primitive_translator(
    primitive_translator: translator.PrimitiveTranslator | None = None, overwrite: bool = False
) -> (
    translator.PrimitiveTranslator
    | Callable[[translator.PrimitiveTranslator], translator.PrimitiveTranslator]
):
    """
    Adds a primitive translator to JaCe's global registry.

    The default set of primitives that are used if nothing is specified to to
    `jace.jit` are stored inside a global registry. To add a translator to this
    registry this function can be used.

    If a translator for `primitive` is already registered an error will be
    generated. However, by specifying `overwrite` `primitive_translator` will
    replace the current one.

    Args:
        primitive_translator: The primitive translator to add to the global registry.
        overwrite: Replace the current primitive translator with `primitive_translator`.

    Note:
        To add a `primitive` property use the `@make_primitive_translator`
        decorator. This function returns `primitive_translator` unmodified,
        which allows it to be used as decorator.
    """

    def wrapper(
        primitive_translator: translator.PrimitiveTranslator,
    ) -> translator.PrimitiveTranslator:
        if primitive_translator.primitive in _PRIMITIVE_TRANSLATORS_REGISTRY and not overwrite:
            raise ValueError(
                f"Explicit override=True needed for primitive '{primitive_translator.primitive}' "
                "to overwrite existing one."
            )
        _PRIMITIVE_TRANSLATORS_REGISTRY[primitive_translator.primitive] = primitive_translator
        return primitive_translator

    return wrapper if primitive_translator is None else wrapper(primitive_translator)


def get_registered_primitive_translators() -> dict[str, translator.PrimitiveTranslator]:
    """
    Returns a copy of the current state of JaCe's global primitive registry.

    The state returned by this function is compatible to what `jace.jit`'s
    `primitive_translators` argument expects. It is important the the returned
    object is decoupled from the registry.
    """
    return _PRIMITIVE_TRANSLATORS_REGISTRY.copy()
