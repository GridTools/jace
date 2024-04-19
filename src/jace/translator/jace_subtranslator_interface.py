# JaCe - JAX jit using DaCe (Data Centric Parallel Programming)
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
    """Interface for all Jax primitive/intrinsic translators.

    A translator for a primitive, sometimes also called intrinsic, translates a single equation of a Jaxpr into its SDFG equivalent.

    A type that implements this interface must fulfil the following properties:
    - It must be stateless.
        It is still possible and explicitly allowed to have an immutable configuration state.
    - All subclasses has to accept '**kwargs' arguments and must forward all unconsumed arguments to the base.
        Thus the '__init__()' function of the base must be called.

    Once a subtranslator is initialized the driver will call its 'get_handled_primitives()' function, which returns the names of all Jax primitives it would like to handle.
    A subtranslator can register for as many primitive it wants.
    At the same time more than one subtranslators can be registered for a single primitive.

    To decide which subtranslator should be used for a single equation the driver will use their 'can_translate_jaxeqn()' function.
    The first subtranslator that returns 'True' will then be used.
    Note it is unspecific if the 'can_translate_jaxeqn()' of the remaining subtranslators is also called.

    There are two ways how to influence the order in which they are processed.
    The first and simple one is to implement 'get_priority()'.
    The driver will order the subtranslators, in ascending order, according to their respective priority.
    Thus the lower the priority the earlier the subtranslator is checked.
    Subtranslators that returns 'JaCeSubTranslatorInterface.DEFAULT_PRIORITY' are handled specially and are _always_ put at the end of the list (in unspecific order).

    The second possibility is to override the '__lt__()' and '__eq__()' functions.
    While this allows more control it might be more difficult.
    If a subtranslator does this, its 'get_priority()' function should return 'NotImplemented'.
    """

    __slots__ = ()

    # Default value for the priority of primitive translators.
    DEFAULT_PRIORITY: Final = int("1" * 64, base=2)

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the interface.

        It is required that subclasses calls this method during initialization.
        """

    def get_handled_primitives(self) -> Collection[str] | str:
        """Returns the names of all Jax primitives that can be handled by this subtranslator.

        The returned collection is used to narrow down which translator should be used to translate a given primitive.
        It is possible that several translators can be registered for the same name.

        See Also:
            'self.can_translate_jaxeqn()' and 'self.get_priority()'.

        Notes:
            It is also possible to return a string instead of a collection with just one element.
        """
        raise NotImplementedError(
            "Class '{type(self).__name__}' does not implement 'get_handled_primitives()'."
        )

    def can_translate_jaxeqn(
        self,
        driver: "JaxprTranslationDriver",
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jcore.JaxprEqn,
    ) -> bool:
        """Tests if 'self' is able to translate the Jax primitive passed as 'eqn'.

        This function is used by the driver translator to determine which subtranslator
        should be used to handle the 'jcore.JaxprEqn', i.e. primitive.
        For a more detailed description of the arguments see 'self.translate_jaxeqn()' function.

        Args:
            driver:         The driver object of the translation.
            in_var_names:   Names of the SDFG variables used as inputs for the primitive.
            out_var_names:  Names of the SDFG variables used as outputs for the primitive.
            eqn:            The 'jcore.JaxprEqn' instance that is currently being handled.

        Notes:
            This function has to consider 'self' and all of its arguments as constant.
            In case there is only one subtranslator registered for a certain primitive,
                it is unspecific if this function will be called before 'self.translate_jaxeqn()' is called.
            This function will never be called for a primitive for which it has not registered itself.
        """
        raise NotImplementedError(
            "Class '{type(self).__name__}' does not implement 'can_translate_jaxeqn()'."
        )

    def translate_jaxeqn(
        self,
        driver: "JaxprTranslationDriver",
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jcore.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        """Translates the Jax primitive into its SDFG equivalent.

        Before the driver will call this function to translate the primitive into an SDFG it will perform the following preparatory tasks:
        - It will allocate the SDFG variables that are used as outputs.
            Their names will be passed through the 'out_var_names' argument, in the same order as 'eqn.outvars'.
        - It will collect the names of the SDFG variables that are used as input and place them in 'in_var_names', in the same order as 'eqn.invars'.
            If an input argument refers to a literal no SDFG variable is created for it and 'None' is passed to indicate this.
        - The driver will create a new terminal state and pass it as 'eqn_state' argument.
            This state is guaranteed to be empty and 'translator.getTerminalState() is eqn_state' holds.

        If 'self' returns 'None' the driver assumes that the whole primitive was constructed inside 'eqn_state', and the terminal state will left unmodified.
        However, in case 'self' explicitly returns a state, the driver will use it as new terminal state.

        Args:
            driver:         The driver object of the translation.
            in_var_names:   List of the names of the arrays created inside the SDFG for the inpts or 'None' in case of a literal.
            out_var_names:  List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The Jax primitive that should be translated.
            eqn_state:      State into which the primitive's SDFG representation should be constructed.

        Notes:
            A subtranslator is free to do anything to the passed 'eqn_state' with the exception of deleting it or modifying any of its _incoming_ interstateedges.
            As a general rule, if the subtranslator has to create more states it should explicitly return the new terminal state.
        """
        raise NotImplementedError(
            "Class '{type(self).__name__}' does not implement 'translate_jaxeqn()'."
        )

    def get_priority(self) -> int:
        """Returns the priority of this translator.

        In case many translators are registered for the same primitive, see 'self.get_handled_primitives()' they must be ordered.
        The translators are ordered, and checked by the driver according to this value.
        The _smaller_ the value the earlier it is checked.

        See Also:
            'self.can_translate_jaxeqn()' and 'self.get_handled_primitives()'.

        Notes:
            By default the function returns 'self.DEFAULT_PRIORITY', which is handled specially, i.e. it is put at the end.
            If a subtranslator opts in to overwrite '__lt__()' instead the function should return 'NotImplemented'.
                Such translators are biased towards lower priorities.
        """
        return self.DEFAULT_PRIORITY

    def has_default_priority(self) -> bool:
        """Checks if 'self' has default priority.

        Notes:
            It is allowed, but not advised to override this function.
                However, it has to be consistent with 'self.get_priority()'.
        """
        try:
            x = self.get_priority()
        except NotImplementedError:
            return False
        if x is NotImplemented:
            return False
        return x is self.DEFAULT_PRIORITY or (x == self.DEFAULT_PRIORITY)

    def __lt__(
        self,
        other: JaCeSubTranslatorInterface,
    ) -> bool:
        """Tests if 'self' should be checked before 'other' in the selection process.

        As outlined in the class description there are two possibilities to influence the order in which subtranslators are checked.
        The simpler one is simply to implement 'get_priority()'.
        The second one, is to override the '__lt__()' function, which allows to inspect the other subtranslators.

        Notes:
            If you override this function it is advised that 'get_priority()' returns 'NotImplemented'.
            This function is never called if either 'self' or 'other' have default priority.
        """
        return self.get_priority() < other.get_priority()

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        """Tests if two subtranslators are equal.

        The default implementation checks if 'self' and 'other' have the same type.
        However, it your subtranslator strongly depend on its configuration you should override this function.

        Notes:
            If you override this function you should also override 'self.__hash__()' to make the two consistent.
        """
        if not isinstance(other, JaCeSubTranslatorInterface):
            return NotImplemented
        return type(self) == type(other)

    def __hash__(self) -> int:
        """Computes the hash of the subtranslator.

        The default implementation return a hash that is based on the class.
        Thus all instances of a particular subtranslator will have the same hash value.

        Notes:
            If you override this function you should also override 'self.__eq__()' to make the two consistent.
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


# end class(JaCeSubTranslatorInterface):
