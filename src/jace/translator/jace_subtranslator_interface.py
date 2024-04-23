# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any

import dace
from jax import core as jcore


if TYPE_CHECKING:
    from .jaxpr_translator_driver import JaxprTranslationDriver


class JaCeSubTranslatorInterface:
    """Interface for all Jax primitive/intrinsic subtranslators.

    A translator for a primitive, sometimes also called intrinsic, translates
    a single equation of a Jaxpr into its SDFG equivalent. A type that
    implements this interface must fulfil the following properties:
    - It must be stateless.
        It is still possible and explicitly allowed to have an
        immutable configuration state.
    - All subclasses has to accept `**kwargs` arguments and must
        forward all unconsumed arguments to the base.

    Subtranslators are rather simple objects that only have to perform
    the translation. The translation process itself is managed by a driver
    object, which owns and manage the subtranslators.
    In the end this implements the delegation pattern.

    A subtranslator uses its `get_handled_primitives()` function to indicate
    for which Jax primitives it want to register. It is important that there
    is no limits on the number of primitives a subtranslator can register itself.
    However, only one subtranslator can be registered for a primitive.
    """

    __slots__ = ()

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the interface.

        It is required that subclasses calls this method during initialization.
        """

    def get_handled_primitives(self) -> Collection[str] | str:
        """Returns the names of all Jax primitives that `self` is able to handle.

        There is no limit on the number of primitives for which a subtranslator
        can register.

        Notes:
            In case a string is returned it is interpreted as 1 element collection.
        """
        raise NotImplementedError(
            "Class '{type(self).__name__}' does not implement 'get_handled_primitives()'."
        )

    def translate_jaxeqn(
        self,
        driver: JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jcore.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        """Translates the Jax primitive into its SDFG equivalent.

        Before the driver calls this function it will perform the following
        preparatory tasks:
        - It will allocate the SDFG variables that are used as outputs.
            Their names will be passed through the `out_var_names` argument,
            in the same order as `eqn.outvars`.
        - It will collect the names of the SDFG variables that are used as input
            and place them in `in_var_names`, in the same order as `eqn.invars`.
            If an input argument refers to a literal no SDFG variable is created
            for it and `None` is passed to indicate this.
        - The subtranslator will create variables that are used as output.
            They are passed as `out_var_names`, same order as in the equation.
        - The driver will create a new terminal state and pass it as
            `eqn_state` argument. This state is guaranteed to be empty and
            `translator.get_terminal_sdfg_state() is eqn_state` holds.

        Then the subtranslator is called. Usually a subtranslator should
        construct the dataflow graph inside `eqn_state`. It is allowed that the
        subtranslators creates more states if needed, but this state machine
        has to have a single terminal state, which must be returned
        and reachable from `eqn_state`.
        If the function returns `None` the driver will assume that
        subtranslator was able to fully construct the dataflow graph
        within `eqn_state`.

        While a subtranslator is forbidden from meddling with the input
        variables mentioned in `in_var_names` in any way, it is allowed to
        modify the output variables. For example he could create a new
        SDFG variable, with different strides. But in that case the
        subtranslator must update the internal mapping of the driver TBA HOW,
        and modify the mapping in `out_var_names`.
        However, the subtranslator is allowed to create internal temporary
        variables. It just have to ensure that no name collision will occur,
        a way to do this is to use a passed variable name as prefix.


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
        raise NotImplementedError(
            "Class '{type(self).__name__}' does not implement 'translate_jaxeqn()'."
        )

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        """Tests if two subtranslators are equal.

        The default implementation checks if `self` and `other` have the same
        type.
        """
        if not isinstance(other, JaCeSubTranslatorInterface):
            return NotImplemented
        return type(self) == type(other)

    def __hash__(self) -> int:
        """Computes the hash of the subtranslator.

        The default implementation return a hash that is based on the class.
        Thus all instances of a particular subtranslator will have the same
        hash value.
        """
        return id(self.__class__)
