# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, Final, final

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
    for which Jax primitives it want to register. It is important that a
    subtranslator can register for as many primitive it wants. At the same
    time, it is possible that multiple subtranslators have registered for a
    single primitive.

    If multiple subtranslator have registered for the same primitive they
    will be ordered by driver. There are two ways how a subtranslator can
    influence this order. The first one is by implementing `get_priority()`,
    the driver will then put them in ascending order.
    I.e. the lower its priority the earlier a subtranslator is checked.
    However, if a subtranslator returns the special value
    `JaCeSubTranslatorInterface.DEFAULT_PRIORITY` it will be always put at the
    end, in unspecific order if multiple translator are involved.

    The second possibility is to override the '__lt__()' function,
    and establish a strict weak order. If a subtranslator overrides this
    function it should also override `get_priority()` to return `NotImplemented`.

    To decide which subtranslator should be used for a specific equation
    the driver will use their 'can_translate_jaxeqn()' function.
    The first subtranslator that returns 'True' will then be used.

    Todo:
        Also come up with a way how to avoid that instances are allowed to access
            some private members of the driver; Possibly by composition.
        Come up with a better way of ordering; maybe introduce fixed priority level.
            And then allows to sort them according to `__lt__()` within the level.
    """

    __slots__ = ()

    # Default value for the priority of primitive translators.
    DEFAULT_PRIORITY: Final = int("1" * 64, base=2)

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
        can register. It is possible that several translators can be registered
        for the same name.

        See Also:
            `self.can_translate_jaxeqn()` and `self.get_priority()`.

        Notes:
            In case a string is returned it is interpreted as 1 element collection.
        """
        raise NotImplementedError(
            "Class '{type(self).__name__}' does not implement 'get_handled_primitives()'."
        )

    def can_translate_jaxeqn(
        self,
        driver: JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jcore.JaxprEqn,
    ) -> bool:
        """Tests if `self` is able to translate the Jax primitive passed as `eqn`.

        This function is used by the driver to determine which of the subtranslators,
        that have registered for a certain type of primitive, should be used.
        For a more detailed description of the arguments see
        `self.translate_jaxeqn()` function.

        Args:
            driver:         The driver object of the translation.
            in_var_names:   Names of the SDFG variables used as inputs for the primitive.
            out_var_names:  Names of the SDFG variables used as outputs for the primitive.
            eqn:            The `jcore.JaxprEqn` instance that is currently being handled.

        Notes:
            In case there is only one subtranslator registered for a certain primitive,
                it is unspecific if this function will be called at all `self.translate_jaxeqn()`.
            This function will never be called for a primitive for which it has not registered itself.
        """
        raise NotImplementedError(
            "Class '{type(self).__name__}' does not implement 'can_translate_jaxeqn()'."
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
        construct the dataflow graph inside it. It is allowed that the
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

    def get_priority(self) -> int:
        """Returns the priority of this translator.

        The value returned by this function is used by the driver to order the
        subtranslators that have registered for the same primitive.
        The _smaller_ the value the earlier it is checked.

        See Also:
            `self.can_translate_jaxeqn()` and `self.get_handled_primitives()`.

        Notes:
            By default the function returns `self.DEFAULT_PRIORITY`, which is
                handled specially, i.e. it is put at the end.
            If a subtranslator instead overrides `__lt__()` this function
                should return `NotImplemented`.
        """
        return self.DEFAULT_PRIORITY

    def has_default_priority(self) -> bool:
        """Checks if `self` has default priority.

        Notes:
            It is allowed, but not advised to override this function.
                However, it has to be consistent with `self.get_priority()`.
        """
        try:
            x = self.get_priority()
        except NotImplementedError:
            return False
        if x is NotImplemented:
            return False
        return x == self.DEFAULT_PRIORITY

    def __lt__(
        self,
        other: JaCeSubTranslatorInterface,
    ) -> bool:
        """Tests if `self` should be checked before `other` in the selection process.

        As outlined in the class description this is the second possibility to
        influence the order of the subtranslator. This function should return
        `True` if `self` should be checked for applicability _before_ `other`.

        Notes:
            If this function is overridden `get_priority()` should return `NotImplemented`.
            This function is never called if either `self` or `other` have default priority.
        """
        return self.get_priority() < other.get_priority()

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        """Tests if two subtranslators are equal.

        The default implementation checks if `self` and `other` have the same
        type. However, if the behaviour of a subtranslator strongly depend on
        its configuration this function should be overridden.

        Notes:
            If you override this function you should also override
                `self.__hash__()` to make the two consistent.
        """
        if not isinstance(other, JaCeSubTranslatorInterface):
            return NotImplemented
        return type(self) == type(other)

    def __hash__(self) -> int:
        """Computes the hash of the subtranslator.

        The default implementation return a hash that is based on the class.
        Thus all instances of a particular subtranslator will have the same
        hash value.

        Notes:
            If you override this function you should also override
                `self.__eq__()` to make the two consistent.
        """
        return id(self.__class__)

    @final
    def __ne__(
        self,
        other: Any,
    ) -> bool:
        return NotImplemented

    @final
    def __le__(
        self,
        other: Any,
    ) -> bool:
        return NotImplemented

    @final
    def __ge__(
        self,
        other: Any,
    ) -> bool:
        return NotImplemented

    @final
    def __gt__(
        self,
        other: Any,
    ) -> bool:
        return NotImplemented
