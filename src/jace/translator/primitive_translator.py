# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Contains the interface for all primitive translators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import dace
from jax import core as jax_core


if TYPE_CHECKING:
    from jace import translator


class PrimitiveTranslatorCallable(Protocol):
    """Callable version of the primitive translators.

    Used for type annotation purposes, classes should be derived from `PrimitiveTranslator` instead.
    You can use `jace.translator.make_primitive_translator()` to add a `primitive` property to
    a callable.
    """

    __slots__ = ()

    @abstractmethod
    def __call__(
        self,
        driver: translator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        """Translates the Jax primitive into its SDFG equivalent.

        Before the driver calls this function it will perform the following
        preparatory tasks:
        - It will allocate the SDFG variables that are used as outputs. Their names will be passed
            through the `out_var_names` argument, in the same order as `eqn.outvars`.
        - It will collect the names of the SDFG variables that are used as input and place them in
            `in_var_names`, in the same order as `eqn.invars`. If an input argument refers to a
            literal no SDFG variable is created for it and `None` is passed to indicate this.
        - The driver will create variables that are used as output. They are passed as
            `out_var_names`, same order as in the equation.
        - The driver will create a new terminal state and pass it as `eqn_state` argument. This
            state is guaranteed to be empty and `translator.terminal_sdfg_state is eqn_state` holds.

        Then the primitive translator is called.
        Usually a primitive translator should construct the dataflow graph inside `eqn_state`.
        It is allowed that the primitive translators creates more states if needed, but this
        state machinery has to have a single terminal state, which must be returned and reachable
        from `eqn_state`. If the function returns `None` the driver will assume that primitive
        translator was able to fully construct the dataflow graph within `eqn_state`.

        While a primitive translator is forbidden from meddling with the input variables mentioned
        in `in_var_names` in any way, it is allowed to modify the output variables. For example
        it could create a new SDFG variable, with different strides. But in that case the primitive
        translator must update the internal mapping of the driver TBA HOW, and modify the mapping
        specified by `out_var_names`. However, the subtranslator is allowed to create internal
        temporary variables. It just have to ensure that no name collision will occur, a way to
        do this is to use a passed variable name as prefix.

        Args:
            driver:         The driver object of the translation.
            in_var_names:   List of the names of the arrays created inside the
                                SDFG for the inpts or `None` in case of a literal.
            out_var_names:  List of the names of the arrays created inside the
                                SDFG for the outputs.
            eqn:            The Jax primitive that should be translated.
            eqn_state:      State into which the primitive`s SDFG representation
                                should be constructed.
        """
        ...


@runtime_checkable
class PrimitiveTranslator(PrimitiveTranslatorCallable, Protocol):
    """Interface for all Jax primitive translators.

    A translator for a primitive translates a single equation of a Jaxpr into its SDFG equivalent.
    For satisfying this interface a concrete implementation must be immutable after construction.

    Primitive translators are simple, but highly specialized objects that are only able to perform
    the translation of a single primitive. The overall translation process itself is managed by a
    driver object, which also owns and manage the primitive translators. In the end this implements
    the delegation pattern.

    You can use `jace.translator.register_primitive_translator()` to register your translator to
    Jace.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def primitive(self) -> str:
        """Returns the name of the Jax primitive that `self` is able to handle."""
        ...
