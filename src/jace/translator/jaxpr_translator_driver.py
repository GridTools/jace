# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import dace
from dace import data as ddata, properties as dprop
from jax import core as jax_core


if TYPE_CHECKING:
    from jace import translator

from jace import util


class JaxprTranslationDriver:
    """Internal driver class for creating an SDFG equivalent of a `Jaxpr` instance.

    The SDFG that is created by this class has a very particular form, which we consider canonical.
    The main feature of a canonical SDFG are:
    - the SDFG is a list of states, ideally each state corresponds to single Jax primitive,
    - all variable names are derived from Jax names,
    - there are only transient variables inside the SDFG,
    - It lacks the special `__return` variable,
    - the `arg_names` parameter is not set.

    For these reasons the SDFG is not directly usable, and further manipulations have to be performed.
    Especially, DaCe's validation function will fail and it is unable to be processed by the optimization pipeline.
    For more information also see `jace.translator.pre_post_translation` module.

    The idea of the translator is extremely simple.
    Since Jaxpr is a list consisting of more or less simple instructions/equations, they get processed one after the other.
    Each equation is translated into its own state that is appended to the SDFG, thus the SDFG is a long list of states.
    In certain cases it might be that an equation needs more states, but this is an exception.

    The actual translation of the equation is not handled by the driver.
    Instead the request is forwarded to a `PrimitiveTranslator` object, also known as subtranslator.
    This is a highly specialized object that is able to handle one kind of primitive.
    For more information on the subtranslators see the documentation of `PrimitiveTranslator`.

    To start a translation the `translate_jaxpr()` function should be called, if this happens it is said that the driver has an ongoing translation.
    If `translate_jaxpr()` is called on a driver that has an ongoing translation, a new translation context will be set up.
    Thus the driver will then translate the supplied (nested) Jaxpr and return the result.
    However, this will have no influence on the translation process that is already going.

    Notes:
        After the main translation has been performed the translator object can be used again.
        Currently the driver will generate only Array as SDFG variables, however, this is a temporary solution.
            For more on that see `add_array()`.
    """

    __slots__ = (
        "_ctx_stack",  # Stack of all contexts
        "_sub_translators",
        "_jax_name_map",
    )

    def __init__(
        self,
        sub_translators: Mapping[str, translator.PrimitiveTranslatorCallable],
    ) -> None:
        """Creates the driver.

        Args:
            sub_translators:    Use these subtranslators to perform the translation.

        Notes:
            `sub_translators` is not copied, thus the user has to guarantee,
                that it will not change during translation.
                It is highly advised but not required to use the output of
                `get_regsitered_primitive_translators()` or pass a copy as argument.
        """

        # Maps the name of a Jax primitive to the primitive translator that should be used.
        #  Note that the subtranslator is only required to be a callable, and immutable.
        #  Allocated through the lifetime of `self`, and shared with the outside.
        self._sub_translators: Mapping[str, translator.PrimitiveTranslatorCallable] = (
            sub_translators
        )

        # Maps Jax variables to the name of its SDFG equivalent.
        #  Note that it is shared among all translation contexts.
        #  This is done to create consistency between SDFG variables
        #  and the names used pretty printed Jaxprs.
        self._jax_name_map: dict[jax_core.Var | util.JaCeVar, str] = {}

        # Context stack and current context.
        #  If it is empty, then no translation process is in process.
        self._ctx_stack: list[translator.TranslatedJaxprSDFG] = []

    def translate_jaxpr(
        self,
        jaxpr: jax_core.ClosedJaxpr,
        *,
        name: str | None = None,
    ) -> translator.TranslatedJaxprSDFG:
        """Perform the translation of a Jaxpr into a SDFG.

        In case this function is called and `self` has an ongoing translation process, a new translation context will be created.
        This means the Jaxpr will be translated independently from the previous one.

        Returns:
            The function will translate the passed Jaxpr object into an SDFG in canonical form.
            This SDFG together with additional meta data, that is needed for further processing is encapsulated inside a `TranslatedJaxprSDFG` object.

        Args:
            name:                   Use this name for the SDFG instead some generated one.
        """
        import jax as _jax

        if len(jaxpr.effects) != 0:
            raise NotImplementedError("'Jaxpr' with side effects are not supported.")
        if not _jax.config.read("jax_enable_x64"):
            # NOTE: What is interesting here is, that the SDFG can be called, but the result is garbage.
            #  Beside that I think it should not work, I think it should not even call,
            #  because of a mismatch in data types.
            #  However, If we work with Jax arrays themselves, it should technically work.
            #  But currently the best we can do, is forbid it!
            raise NotImplementedError(
                "You have disabled 'x64' support in Jax, which interferes with the calling of the SDFG. "
                "SDFG generated in this way will fail to call."
            )

        # NOTE: If `self` is already allocated, i.e. has an ongoing translation process,
        #       the `_allocate_translation_ctx()` function will start a new context.
        #       Thus the driver will start to translate a second (nested) SDFG.
        #       Also note that there is no mechanism that forces the integration of the nested SDFG/Jaxpr,
        #       this must be done manually.
        self._allocate_translation_ctx(
            name=name,
        )
        self._create_constants(
            jaxpr=jaxpr,
        )
        self._create_initial_input(jaxpr=jaxpr)
        # Note that `self` and `jsdfg` still share the same underlying memory, i.e. context.
        jsdfg: translator.TranslatedJaxprSDFG = self._translate_jaxpr_internal(jaxpr)
        self._clear_translation_ctx()

        return jsdfg

    def append_new_state(
        self,
        label: str | None = None,
        condition: dprop.CodeBlock | None = None,
        assignments: Mapping[str, Any] | None = None,
        prev_state: dace.SDFGState | None = None,
    ) -> dace.SDFGState:
        """Creates a new `SDFGState` and adds it to the SDFG.

        By default the new state is appended to the current terminal state,
        which will also update the terminal state of recorded inside `self`.

        However, if `prev_state` is specified the state new state will be appended to `prev_state` instead.
        The terminal state of `self` will only be modified if `prev_state` is the current terminal state.

        Args:
            label:          The name that should be given to the new `SDFGState`.
            condition:      The condition of the state transitions used on the `InterstateEdge`.
            assignments:    Symbol assignments that should be done during the transition.
            prev_state:     Alternative `SDFGState` at which we should append the new state.
        """
        if isinstance(label, str) and (not util.VALID_SDFG_OBJ_NAME.fullmatch(label)):
            raise ValueError(f"Can not create state with label '{label}' since it is invalid.")

        # Decide if appending to that state will modify the terminal state.
        modify_term_state: bool = False
        if (prev_state is self._ctx.terminal_state) or (prev_state is None):
            modify_term_state = True
            app_state = self._ctx.terminal_state
        else:
            app_state = prev_state

        new_state = self._ctx.sdfg.add_state(label, is_start_block=False)
        self._ctx.sdfg.add_edge(
            app_state,
            new_state,
            dace.sdfg.InterstateEdge(condition=condition, assignments=assignments),
        )

        if modify_term_state:
            self._ctx.terminal_state = new_state
        return new_state

    @property
    def arrays(self) -> Mapping[str, ddata.Data]:
        """Get all data descriptors that are currently known to the SDFG.

        Notes:
            Essentially a shorthand and preferred way for `self.sdfg.arrays`.
            For getting a specific data descriptor use `self.get_array()`.
        """
        return cast(Mapping[str, ddata.Data], self._ctx.sdfg.arrays)

    def get_array(
        self,
        name: str | jax_core.Atom | util.JaCeVar,
    ) -> ddata.Data:
        """Returns the SDFG `Data` object `name` referees to.

        If `name` is a string it is directly interpreted as the name of an SDFG variable.
        In other cases it is first translated using `self.map_jax_var_to_sdfg()`.
        """
        if isinstance(name, (jax_core.Var, util.JaCeVar)):
            sdfg_name: str = self.map_jax_var_to_sdfg(name)
        elif isinstance(name, str):
            sdfg_name = name
        else:
            raise TypeError(f"The literal '{name}' does not have an SDFG equivalent.")
        if sdfg_name not in self._ctx.sdfg.arrays:
            raise KeyError(f"Requested SDFG object '{name}' is not known.")
        return self._ctx.sdfg.arrays[sdfg_name]

    @overload
    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jax_core.Atom | util.JaCeVar,
    ) -> str: ...

    @overload
    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jax_core.Atom | util.JaCeVar,
        allow_fail: Literal[True],
    ) -> str | None: ...

    def map_jax_var_to_sdfg(
        self,
        jax_var: jax_core.Atom | util.JaCeVar,
        allow_fail: bool = False,
    ) -> str | None:
        """Get the _name_ of the SDFG variable to which `jax_var` is referring to.

        Args:
            jax_var:        The Jax variable to look up.
            allow_fail:     If mapping is not known return `None` instead of raising `KeyError`.
        """
        if isinstance(jax_var, jax_core.Literal):
            raise RuntimeError("There is no SDFG variable for literal '{jax_var}'.")
        if jax_var in self._jax_name_map:
            sdfg_name = self._jax_name_map[jax_var]
        elif allow_fail:
            return None
        else:
            KeyError(f"The Jax variable '{jax_var}' was never registered.")
        if sdfg_name not in self._ctx.sdfg.arrays:
            raise KeyError(
                f"Jax variable '{jax_var}' was supposed to map to '{sdfg_name}',"
                " but no such SDFG variable is known."
            )
        return sdfg_name

    @property
    def sdfg(self) -> dace.SDFG:
        """Returns the SDFG that is currently constructed.

        If you want access to the arrays of the SDFG use `self.arrays()`/`self.get_array()`.
        """
        return self._ctx.sdfg

    def is_allocated(self) -> bool:
        """Tests if `self` has an allocated context.

        If `self` is allocated then there is also an ongoing translation process.
        """
        if len(self._ctx_stack) != 0:
            return True
        return False

    def is_root_translator(self) -> bool:
        """Tests if `self` is a root translator.

        The root translator (context) is the very first translator process that was started.
        """
        if not self.is_allocated():
            raise RuntimeError("Driver is not allocated.")
        if len(self._ctx_stack) == 1:
            return True
        return False

    def add_jax_name_mapping(
        self,
        jax_var: jax_core.Var | util.JaCeVar,
        sdfg_name: str,
    ) -> JaxprTranslationDriver:
        """Creates a new mapping between `jax_var` to `sdfg_name`.

        If the mapping already exists an error will be generated.
        This function is not able to delete a variable mapping that was established before, for this use TBA.

        Args:
            jax_var:     The Jax variable.
            sdfg_name:   The name of the corresponding SDFG variable.
        """
        assert len(sdfg_name) > 0

        if jax_var in self._jax_name_map:
            raise ValueError(
                f"Tried to create the mapping '{jax_var} -> {sdfg_name}', but the variable is already mapped."
            )
        if sdfg_name not in self._ctx.sdfg.arrays:
            raise KeyError(f"Mapping '{jax_var} -> {sdfg_name}': SDFG target unknown.")
        if sdfg_name in util.FORBIDDEN_SDFG_VAR_NAMES:
            raise NameError(f"Mapping '{jax_var} -> {sdfg_name}': Forbidden name.")

        self._jax_name_map[jax_var] = sdfg_name
        return self

    def add_array(
        self,
        arg: jax_core.Atom | util.JaCeVar,
        *,
        name_prefix: str | None = None,
        update_var_mapping: bool = False,
    ) -> str:
        """Creates an SDFG variable for the Jax variable `arg` and returns its SDFG name.

        The SDFG object is always created as a transient.

        By default the function will use `jace.util.propose_jax_name()` to derive the name that should be used.
        However, by passing a `JaCeVar` with a name it is possible to suggest a specific name.
        In addition it is possible to specify `name_prefix` to prefix name that would be used.

        The function will not update the internal variable mapping.
        If this is desired one can set `update_var_mapping`, for forcing this.

        Args:
            arg:                The Jax object for which a SDFG equivalent should be created.
            name_prefix:        If given it will be used as prefix for the name.
            update_var_mapping: Update the internal variable mapping; by default `False`.

        Notes:
            Currently the function will always create an Array, even if the Jax variable refers to a scalar.
                This is done to work around some difficulties with scalar return values and so on.
                This issue should actually handled in the post processing stage, but currently it is not.
                However, from a point of building an SDFG manually, there is no difference between a Scalar and an Array.
                According to the dace developer, the majority of the backend, i.e. optimization pipeline, should be handle to handle it.
                But there are some special parts that might explicitly want a scalar, it also might block certain compiler optimization.
        """
        shape: tuple[int | dace.symbol | str, ...] = util.get_jax_var_shape(arg)
        dtype: dace.typeclass = util.get_jax_var_dtype(arg)
        storage: dace.StorageType = dace.StorageType.Default  # Set at later stages (optimization)
        offset = None
        as_transient = True
        strides = None

        if shape == ():  # Shape of a DaCe scalar.
            shape = (1,)

        # Propose a name and if needed extend it.
        arg_name = util.propose_jax_name(arg, self._jax_name_map)
        if name_prefix is not None:
            if not util.VALID_SDFG_VAR_NAME.fullmatch(name_prefix):
                raise ValueError(f"add_array({arg}): Supplied invalid prefix '{name_prefix}'.")
            arg_name = f"{name_prefix}{arg_name}"

        # final checks
        if arg_name in self._ctx.sdfg.arrays:
            raise ValueError(f"add_array({arg}): The proposed name '{arg_name}', is used.")

        self._ctx.sdfg.add_array(
            name=arg_name,
            shape=shape,
            strides=strides,
            offset=offset,
            storage=storage,
            dtype=dtype,
            transient=as_transient,
        )

        if update_var_mapping:
            try:
                # If the mapping fails, remove the variable from the SDFG.
                self.add_jax_name_mapping(jax_var=arg, sdfg_name=arg_name)
            except:
                del self._ctx.sdfg.arrays[arg_name]
                raise

        return arg_name

    @overload
    def create_jax_var_list(
        self,
        jax_var_list: Sequence[jax_core.Atom | util.JaCeVar],
        prevent_creation: bool = False,
        only_creation: bool = True,
        handle_literals: bool = False,
        **kwargs: Any,
    ) -> list[str]: ...

    @overload
    def create_jax_var_list(  # type: ignore[misc]
        self,
        jax_var_list: Sequence[jax_core.Atom | util.JaCeVar],
        prevent_creation: bool = False,
        only_creation: bool = False,
        handle_literals: bool = False,
        **kwargs: Any,
    ) -> list[None | str]: ...

    def create_jax_var_list(  # type: ignore[misc]
        self,
        jax_var_list: Sequence[jax_core.Atom | util.JaCeVar],
        prevent_creation: bool = False,
        only_creation: bool = False,
        handle_literals: bool = False,
        **kwargs: Any,
    ) -> list[None | str]:
        """Creates SDFG variables for the listed Jax variables and returns their SDFG names.

        If a Jax variable already has a SDFG equivalent then the function will use this variable.
        If no corresponding SDFG variable is known the function will create one using `add_array()`.

        By setting `prevent_creation` the function will not create any new SDFG variables,
        if no corresponding SDFG variable exists an error is generated.
        By setting `only_creation` the function will only create new SDFG variables,
        if a variable already have a corresponding SDFG variable an error will be created.

        By default literals cause an error.
        However, by setting `handle_literals` to `True` literals will will be included in the output with the value `None`.

        Args:
            jax_var_list:       The list of Jax variables that should be transformed to SDFG names.
            prevent_creation:   Never create a variable, all must already be known.
            only_creation:      Always create a variable, it is an error if one already exist.
            handle_literals:    Allow the processing of literals.
            kwargs:             Will be forwarded to `self.add_array()` in case a variable is created.

        Todo:
            Rollback if the creation fails.
        """
        if only_creation and prevent_creation:
            raise ValueError("Specified both 'only_creation' and 'prevent_creation'.")

        ret_list: list[None | str] = []
        for jax_var in jax_var_list:
            if isinstance(jax_var, jax_core.Literal):
                if not handle_literals:
                    raise ValueError("Encountered a literal but `handle_literals` was `False`.")
                sdfg_name = None
            else:
                mapped_sdfg_name: str | None = self.map_jax_var_to_sdfg(jax_var, allow_fail=True)
                if prevent_creation and (mapped_sdfg_name is None):
                    raise ValueError(f"'prevent_creation' given but have to create '{jax_var}'.")
                if mapped_sdfg_name is None:
                    sdfg_name = self.add_array(arg=jax_var, **kwargs)
                elif only_creation:
                    raise ValueError(f"'only_creation' given '{jax_var}' already exists.")
                else:
                    sdfg_name = mapped_sdfg_name
            ret_list.append(sdfg_name)

        return ret_list

    def _create_initial_input(
        self,
        jaxpr: jax_core.ClosedJaxpr,
    ) -> Sequence[str]:
        """This function will create the internal input variables that are used for the SDFG.

        Args:
            jaxpr:                  The Jaxpr that we want to translate.

        Returns:
            The list of SDFG variables used as input arguments of `jaxpr` in the same order.

        Notes:
            The function will populate the `inp_names` member of the current context.
        """
        if not self.is_allocated():
            raise RuntimeError("Driver is not allocated, can not create constants.")
        assert len(self._ctx.inp_names) == 0

        # Handle the initial input arguments
        init_in_var_names: Sequence[str] = self.create_jax_var_list(
            jax_var_list=jaxpr.jaxpr.invars,
            only_creation=True,  # Nothing exists yet.
            handle_literals=False,  # Initial arguments are never literals
            update_var_mapping=True,
        )
        # This forces the code to only accept kwargs; it is also part of "what a canonical sdfg" is.
        self.sdfg.arg_names = []

        # The output list is populated by `self._translate_jaxpr_internal()`
        self._ctx.inp_names = tuple(init_in_var_names)

        return init_in_var_names

    def _create_constants(
        self,
        jaxpr: jax_core.ClosedJaxpr,
    ) -> Sequence[str]:
        """Creates all constants requested by the `jaxpr`.

        The function will create an SDFG variable and add them as constant to the SDFG.
        The value they should have is deepcopied.

        Returns:
            Names of the SDFG variables created for the constants in the same order.
        """
        from copy import deepcopy

        if not self.is_allocated():
            raise RuntimeError("Driver is not allocated, can not create constants.")
        if len(jaxpr.consts) == 0:
            return ()

        sdfg_const_names: Sequence[str] = self.create_jax_var_list(
            jax_var_list=jaxpr.jaxpr.constvars,
            only_creation=True,  # Nothing exists yet.
            handle_literals=False,  # It seems that constants are never literals.
            name_prefix="__const_",
            update_var_mapping=True,
        )
        for sdfg_name, const_value in zip(sdfg_const_names, jaxpr.consts, strict=True):
            self._ctx.sdfg.add_constant(
                sdfg_name, deepcopy(const_value), self._ctx.sdfg.arrays[sdfg_name]
            )
        return sdfg_const_names

    def _allocate_translation_ctx(
        self,
        name: str | None = None,
    ) -> JaxprTranslationDriver:
        """This function allocates and initialize the members of the translation context of `self`.

        If this function is called and `self` is already allocated, the function will create a new context.
        This allows the driver to handle nested Jaxpr.
        The first context that is created is also known as root translator.

        Args:
            name:               The name of the SDFG.
        """
        from jace import translator  # Cyclic import

        # Create a new translation context and put it on the stack.
        self._ctx_stack.append(
            translator.TranslatedJaxprSDFG(
                name=name,
            )
        )

        if self.is_root_translator():
            # In the future we will populate the generate state here, i.e. if we are on GPU or not and so on.
            assert len(self._jax_name_map) == 0

        return self

    @property
    def _ctx(self) -> translator.TranslatedJaxprSDFG:
        """Returns the currently active translation context."""
        assert len(self._ctx_stack) != 0, "No context is active."
        return self._ctx_stack[-1]

    def _clear_translation_ctx(self) -> JaxprTranslationDriver:
        """This function deallocate the currently active translation context of `self`.

        Notes:
            While it is allowed for outside code to call this function explicit it is is most likely an error.
            If `self` is not allocated this function acts as a noops.
            If `self` is a root translator, then the function will also deallocate the shared state of `self`.
        """
        if not self.is_allocated():
            return self

        if self.is_root_translator():
            # The translation as a whole has finished, so restore the driver,
            #  i.e. delete all the shared state.
            self._jax_name_map = {}

        # Remove the current head stack.
        _ = self._ctx_stack.pop()
        return self

    def _translate_single_eqn(
        self,
        eqn: jax_core.JaxprEqn,
    ) -> tuple[Sequence[str | None], Sequence[str]]:
        """Translate `eqn` into its SDFG equivalent.

        To do this the function will do the following steps:
        - Assemble the in and output variables.
        - Select the appropriate subtranslator to use.
        - Create a new empty state terminal state.
        - Call the subtranslator to perform the translation inside the new state.

        Returns:
            The SDFG names that were used as input and output are returned.
            The inputs might contain `None` which indicates that that input was a Jax Literal.

        Notes:
            The equation, `eqn` must come from the unclosed jaxpr instance.
            The function will perform some consistency checking after the subtranslator was called.
        """
        if len(eqn.effects) != 0:
            raise NotImplementedError(f"Equation '{eqn}' has side effects.")

        # Input/Output variables
        #  Using a tuple for the input ensures that it cannot be modified.
        in_var_names: Sequence[str | None] = tuple(
            self.create_jax_var_list(
                eqn.invars,
                prevent_creation=True,  # Inputs must already exists.
                handle_literals=True,  #   but they can be literals.
            )
        )
        out_var_names: MutableSequence[str] = self.create_jax_var_list(
            eqn.outvars,
            only_creation=True,  # Output must not exist yet.
            update_var_mapping=True,
        )

        # Find the subtranslator
        prim_name: str = eqn.primitive.name
        if prim_name not in self._sub_translators:
            raise NotImplementedError(f"No subtranslators known to handle '{prim_name}'.")
        subtranslator = self._sub_translators[prim_name]

        # Create the state into which the equation should be translated
        last_term_state: dace.SDFGState = self._terminal_sdfg_state  # noqa: F841 # Will be used later
        eqn_state = self.append_new_state(
            label=f"{eqn.primitive.name}_{'_'.join(out_var_names)}",
            prev_state=None,  # forces terminal state to use
        )

        # Now perform the actual translation of the equation.
        new_sdfg_term_state = subtranslator(
            driver=self,
            in_var_names=in_var_names,
            out_var_names=out_var_names,  # Might be modified by the subtranslator!
            eqn=eqn,
            eqn_state=eqn_state,
        )

        # Determine the new (tentative) terminal state of the SDFG we are building.
        if new_sdfg_term_state is None:
            if eqn_state is not self._ctx.terminal_state:
                raise RuntimeError("Inconsistent terminal state was detected.")
            new_sdfg_term_state = eqn_state

        # In case a subtranslator decided to not use the variables we created for it, which is allowed
        #  but it must update the `out_var_names` list correctly, we will now verify this.
        for expectedSDFGName, jax_var in zip(out_var_names, eqn.outvars, strict=True):
            mapped_sdfg_name = self.map_jax_var_to_sdfg(jax_var)
            if mapped_sdfg_name != expectedSDFGName:
                raise ValueError(
                    f"Mapping inconsistency detected, expected that Jax variable"
                    f" '{jax_var}' maps to '{expectedSDFGName}' but it actually"
                    f" maps to '{mapped_sdfg_name}'."
                )

        # Modify terminal root state of 'self'
        self._ctx.terminal_state = new_sdfg_term_state

        return (in_var_names, out_var_names)

    def _translate_jaxpr_internal(
        self,
        jaxpr: jax_core.ClosedJaxpr,
    ) -> translator.TranslatedJaxprSDFG:
        """Performs the actual translation of the Jaxpr into an SDFG.

        The function assumes that the context is allocated as well as the initial variables.
        The function will return the internal state of `self` encapsulated inside a `TranslatedJaxprSDFG` object.
        However, it will not deallocate the translation context, thus `self` and the return value share the same memory.

        Args:
            jaxpr:      The Jaxpr to translate.

        Notes:
            The function will unconditionally handle empty Jaxpr.
            Equations that store into drop variables, i.e. with name `_`, will be skipped.
                Jax used such variables to indicate that it is not needed, transformations such as `grad` include them.
        """
        nb_translated_eqn: int = 0
        out_var_names: Sequence[str] = ()

        # Translate the equations one by one.
        for eqn in jaxpr.jaxpr.eqns:
            if any(util.is_drop_var(outVar) for outVar in eqn.outvars):
                assert all(util.is_drop_var(outVar) for outVar in eqn.outvars)
                continue
            _, out_var_names = self._translate_single_eqn(eqn=eqn)
            nb_translated_eqn += 1

        # There were no equation, so handle the copying of input to output.
        if nb_translated_eqn == 0:
            out_var_names = self._handle_null_jaxpr(jaxpr)

        # Set the output names inside the context.
        self._ctx.out_names = tuple(out_var_names)

        return self._ctx

    def _handle_null_jaxpr(
        self,
        jaxpr: jax_core.ClosedJaxpr,
    ) -> Sequence[str]:
        """This function is called in case a `Jaxpr` with zero equations is encountered.

        A function with zero equation might still have output, in which case an input is copied to an output.
        This function will handle the copying from the input into the corresponding output variable.
        It is important that the function will remove the input and output variables from the internal mapping.

        Returns:
            The function returns a list denoting the SDFG variables that refers to the output.
            The order of the list is the same as in `jaxpr.jaxpr.outvars`.
        """
        assert self._ctx.terminal_state is self._ctx.start_state
        assert len(self._ctx.inp_names) > 0
        assert len(self._ctx.out_names) == 0

        # There is not output so we do not have to copy anything around.
        if len(jaxpr.out_avals) == 0:
            return ()

        # List of the output variables.
        out_var_names: list[str] = []

        # If we are here then we are dealing with a nested SDFG/Jaxpr, that has output.
        #  Because an input also serves as output, the nested SDFG will have a connector for the
        #  input and one for the output, but both with the same name.
        #  This will make node validation fail.
        #  We have to work around this by introducing some fake copies, which will be removed by DaCe later.
        for jax_out_var in jaxpr.jaxpr.outvars:
            # Since the output is also used as an input the variable mapping must be already known.
            sdfg_in_name: str = self.map_jax_var_to_sdfg(jax_out_var)

            # Now we create a variable that serves as true output, however, since the Jax variable
            #  is already known we can not update the variable mapping.
            sdfg_out_name = self.add_array(
                jax_out_var,
                name_prefix="_zero_equation_output_for_",
                update_var_mapping=False,
            )
            out_var_names.append(sdfg_out_name)

            # Now we perform the copy from the input variable in the newly created output variable.
            inp_acc = self._start_state.add_read(sdfg_in_name)
            out_acc = self._start_state.add_write(sdfg_out_name)
            self._start_state.add_nedge(
                src=inp_acc,
                dst=out_acc,
                data=dace.Memlet.from_array(sdfg_in_name, self.get_array(sdfg_in_name)),
            )

            # A Jax variable now has, in some sense, two SDFG equivalent, the input, that was previously created by
            #  `self._create_initial_input()` and the `sdfg_out_name` we just created.
            #  But we can not add this to the mapping, because of this situation we will now remove the variable from the mapping all together.
            #  I am open for different approaches.
            #  Note that input variables that are not used as outputs, will remain in the mapping.
            self._jax_name_map.pop(jax_out_var)

        return tuple(out_var_names)

    @property
    def _start_state(self) -> dace.SDFGState:
        return cast(dace.SDFGState, self._ctx.start_state)

    @property
    def _terminal_sdfg_state(self) -> dace.SDFGState:
        """Returns the current terminal state of the SDFG under construction."""
        return cast(dace.SDFGState, self._ctx.terminal_state)
