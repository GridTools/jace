# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the translator that actually builds an SDFG based on a Jaxpr description."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import dace
from dace import data as ddata, properties as dprop
from jax import core as jax_core

from jace import util


if TYPE_CHECKING:
    from jace import translator


class JaxprTranslationBuilder:
    """Internal builder class for creating an SDFG equivalent of a `Jaxpr` instance.

    The SDFG created by this class has a very particular form, which we call
    canonical. The main features of such an SDFG are:
    - the SDFG is a list of states, ideally each state corresponds to single Jax primitive,
    - it has a single source and sink state.
    - all variable names are derived from Jax names,
    - there are only transient variables inside the SDFG,
    - It lacks the special `__return` variable,
    - the `arg_names` parameter is not set.

    For these reasons the SDFG is not directly usable, and further manipulations
    have to be performed. Especially, DaCe's validation function will fail and
    it is unable to be processed by JaCe's optimization pipeline. For more
    information also see `jace.translator.post_translation` module.

    The idea of the translator is extremely simple. A Jaxpr is essentially a
    list consisting of more or less simple instructions/equations, they get
    processed one after the other. Each equation is translated into its own
    state that is successively appended to the SDFG, while the SDFG is being
    build, which explains the particular form of the SDFG.

    However, the actual translation of the equations is not handled by the
    builder. Instead the request is forwarded to a `PrimitiveTranslator`
    object, known as primitive translator. This is a highly specialized object
    that is able to handle one kind of primitive. For more information on them
    see the documentation of `PrimitiveTranslator`.

    To start a translation the `translate_jaxpr()` function has to be called,
    if this happens it is said that the builder has an ongoing translation.
    The first translator is known as root, translator. If `translate_jaxpr()`
    is called on a builder that has an ongoing translation, a new translation
    context will be set up. Thus the builder will then translate the supplied
    (nested) Jaxpr and return the result. However, this will have no influence
    on the translation process that is already going.

    Args:
        primitive_translators: Primitive translators to use in the translation.

    Notes:
        After a translation has been performed the translator object can be used
        again. Currently the builder will generate only Array as SDFG variables,
        however, this is a temporary solution, see `add_array()`.
    """

    _primitive_translators: Mapping[str, translator.PrimitiveTranslatorCallable]
    _jax_name_map: dict[jax_core.Var | util.JaCeVar, str]
    _ctx_stack: list[TranslationContext]

    def __init__(
        self, primitive_translators: Mapping[str, translator.PrimitiveTranslatorCallable]
    ) -> None:
        # Maps name of primitives to the associated translator.
        self._primitive_translators = {**primitive_translators}

        # Maps Jax variables to the name of its SDFG equivalent.
        #  Shared between all translation contexts, to ensure consecutive variable naming as
        #  seen as in a pretty printed Jaxpr.
        #  Will be cleared by `_clear_translation_ctx()` at the end of the root translation.
        self._jax_name_map = {}

        # Stack of all context, to handle nested Jaxpr instances.
        #  The first one, i.e. index 0, is known as head translator.
        self._ctx_stack = []

    def translate_jaxpr(
        self, jaxpr: jax_core.ClosedJaxpr, *, name: str | None = None
    ) -> TranslationContext:
        """Perform the translation of a Jaxpr into a SDFG.

        In case this function is called and `self` has an ongoing translation
        process, a new translation context will be created. This allows to
        handle nested Jaxprs. However, the variable map is shared among all.

        Returns:
            The function will translate the passed Jaxpr object into an SDFG
            in canonical form. This SDFG together with additional meta data,
            that is needed for further processing is encapsulated inside a
            `TranslationContext` object. For further use it should be passed
            to `postprocess_jaxpr_sdfg()`.

        Args:
            name: Use this name for the SDFG instead some generated one.
        """

        if len(jaxpr.effects) != 0:
            raise NotImplementedError("'Jaxpr' with side effects are not supported.")

        # NOTE: If `self` is already allocated, i.e. has an ongoing translation process,
        #       the `_allocate_translation_ctx()` function will start a new context.
        #       Thus the builder will start to translate a second (nested) SDFG.
        #       Also note that there is no mechanism that forces the integration of the nested
        #       SDFG/Jaxpr, this must be done manually.
        self._allocate_translation_ctx(name=name)
        self._create_constants(jaxpr=jaxpr)
        self._create_initial_input(jaxpr=jaxpr)

        return self._translate_jaxpr_internal(jaxpr)

    def append_new_state(
        self,
        label: str | None = None,
        condition: dprop.CodeBlock | None = None,
        assignments: Mapping[str, Any] | None = None,
        prev_state: dace.SDFGState | None = None,
    ) -> dace.SDFGState:
        """Creates a new `SDFGState`, adds it to the SDFG and returns it.

        By default the new state is appended to the current terminal state.
        However, if `prev_state` is specified it will be appended to it. In
        case the new state is appended to the current terminal state, this will
        modify the terminal state of `self`.

        Args:
            label: The name that should be given to the new `SDFGState`.
            condition: The condition of the state transitions used on the `InterstateEdge`.
            assignments: Symbol assignments that should be done during the transition.
            prev_state: Alternative `SDFGState` at which we should append the new state.

        Notes:
            It is potentially dangerous to not append to the current terminal
            state, as a canonical SDFG only has one sink state. If this is done
            the user has to ensure, that at the end of the processing the SDFG
            is back in canonical form.
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

    def get_array(self, name: str | jax_core.Atom | util.JaCeVar) -> ddata.Data:
        """Returns the SDFG `Data` object `name` referees to.

        `name` can either be a string, in which case it is interpreted as a
        verbatim SDFG name. If it is a Jax or JaCe variable, the function will
        first perform a lookup using `self.map_jax_var_to_sdfg(name)`.
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
        self, jax_var: jax_core.Atom | util.JaCeVar, allow_fail: Literal[False] = False
    ) -> str: ...

    @overload
    def map_jax_var_to_sdfg(
        self, jax_var: jax_core.Atom | util.JaCeVar, allow_fail: Literal[True]
    ) -> str | None: ...

    def map_jax_var_to_sdfg(
        self, jax_var: jax_core.Atom | util.JaCeVar, allow_fail: bool = False
    ) -> str | None:
        """Get the name of the SDFG variable to which `jax_var` is referring to.

        Args:
            jax_var: The Jax variable to look up.
            allow_fail: If no mapping is known return `None` instead of raising `KeyError`.
        """
        if isinstance(jax_var, jax_core.Literal):
            raise RuntimeError(f"There is no SDFG variable for literal '{jax_var}'.")
        if jax_var in self._jax_name_map:
            sdfg_name = self._jax_name_map[jax_var]
        elif allow_fail:
            return None
        else:
            raise KeyError(f"The Jax variable '{jax_var}' was never registered.")
        if sdfg_name not in self._ctx.sdfg.arrays:
            raise KeyError(
                f"Jax variable '{jax_var}' was supposed to map to '{sdfg_name}',"
                " but no such SDFG variable is known."
            )
        return sdfg_name

    @property
    def sdfg(self) -> dace.SDFG:
        """Returns the SDFG that is currently constructed."""
        return self._ctx.sdfg

    def is_allocated(self) -> bool:
        """Tests if `self` has an allocated context.

        If `self` is allocated then there is also an ongoing translation process.
        """
        return len(self._ctx_stack) != 0

    def is_root_translator(self) -> bool:
        """Tests if `self` is the root translator.

        The root translator (context) is the very first translator process.
        """
        if not self.is_allocated():
            raise RuntimeError("Builder is not allocated.")
        return len(self._ctx_stack) == 1

    def add_jax_name_mapping(
        self, jax_var: jax_core.Var | util.JaCeVar, sdfg_name: str
    ) -> JaxprTranslationBuilder:
        """Creates a new mapping between `jax_var` to `sdfg_name`.

        If the mapping already exists an error will be generated. This function
        is not able to delete a variable mapping that was established before.

        Args:
            jax_var: The Jax variable.
            sdfg_name: The name of the corresponding SDFG variable.
        """
        assert sdfg_name

        if jax_var in self._jax_name_map:
            raise ValueError(
                f"Cannot change the mapping of '{jax_var}' from"
                f" '{self.map_jax_var_to_sdfg(jax_var)}' to '{sdfg_name}'."
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
        """Creates an SDFG variable for Jax variable `arg` and returns its SDFG name.

        The SDFG object is always created as a transient. Furthermore, the
        function will not update the internal variable mapping, by default.

        By default the function will use `jace.util.propose_jax_name()` to derive
        the name that should be used. However, by passing a `JaCeVar` with a
        name it is possible to suggest a specific name. In addition it is possible
        to specify `name_prefix` to supply a prefix to the determined name that
        should be used.

        Args:
            arg: The Jax object for which a SDFG equivalent should be created.
            name_prefix: If given it will be used as prefix for the name.
            update_var_mapping: Update the internal variable mapping; by default `False`.

        Notes:
            As a temporary fix for handling scalar return values, the function
            will always generate arrays, even if `arg` is a scalar. According to
            the DaCe developer, the majority of the backend, i.e. optimization
            pipeline, should be able to handle it. But there are some special
            parts that might explicitly want a scalar, it also might block
            certain compiler optimization.
        """

        if isinstance(arg, jax_core.Literal):
            raise ValueError(f"Can not generate an SDFG variable for literal '{arg}'.")

        shape: tuple[int | dace.symbol | str, ...] = util.get_jax_var_shape(arg)
        dtype: dace.typeclass = util.get_jax_var_dtype(arg)
        storage: dace.StorageType = dace.StorageType.Default  # Set at later stages (optimization)
        offset = None
        as_transient = True
        strides = None

        # Temporary fix for handling DaCe scalars, see above for more.
        shape = shape or (1,)

        # Propose a name and if needed extend it.
        arg_name = util.propose_jax_name(arg, self._jax_name_map)
        if name_prefix:
            arg_name = f"{name_prefix}{arg_name}"

        # final checks
        if arg_name in self._ctx.sdfg.arrays:
            raise ValueError(f"add_array({arg}): The proposed name '{arg_name}', is used.")
        if not util.VALID_SDFG_VAR_NAME.fullmatch(arg_name):
            raise ValueError(f"add_array({arg}): The proposed name '{arg_name}', is invalid.")
        if arg_name in util.FORBIDDEN_SDFG_VAR_NAMES:
            raise ValueError(f"add_array({arg}): The proposed name '{arg_name}', is forbidden.")

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

        If a Jax variable already has a SDFG equivalent then the function will
        use this variable. If no corresponding SDFG variable is known the function
        will create one using `add_array()`.

        By setting `prevent_creation` the function will not create any new SDFG
        variables, if no corresponding SDFG variable exists an error is generated.
        By setting `only_creation` the function will only create new SDFG variables,
        if a variable already have a corresponding SDFG variable an error will be
        generated.

        By default literals cause an error. However, by setting `handle_literals`
        to `True` literals will will be included in the output with the value `None`.

        Args:
            jax_var_list: The list of Jax variables that should be transformed to SDFG names.
            prevent_creation: Never create a variable, all must already be known.
            only_creation: Always create a variable, it is an error if one already exist.
            handle_literals: Allow the processing of literals.
            kwargs: Will be forwarded to `self.add_array()` if a variable is created.

        Todo:
            - Rollback if the creation fails.
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
                    raise ValueError(f"'only_creation' given but '{jax_var}' already exists.")
                else:
                    sdfg_name = mapped_sdfg_name
            ret_list.append(sdfg_name)

        return ret_list

    def _create_initial_input(self, jaxpr: jax_core.ClosedJaxpr) -> None:
        """Creates the input variables of `jaxpr`.

        Notes:
            The function will populate the `inp_names` member of the current context.
        """
        assert self.is_allocated(), "Builder is not allocated, can not create constants."
        assert self._ctx.inp_names is None

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

    def _create_constants(self, jaxpr: jax_core.ClosedJaxpr) -> None:
        """Creates all constants requested by the `jaxpr`.

        The function will create an SDFG variable and add them as constant to
        the SDFG. Their value is deepcopied.
        """
        assert self.is_allocated(), "Builder is not allocated, can not create constants."
        if len(jaxpr.consts) == 0:
            return

        sdfg_const_names: Sequence[str] = self.create_jax_var_list(
            jax_var_list=jaxpr.jaxpr.constvars,
            only_creation=True,  # Nothing exists yet.
            handle_literals=False,  # It seems that constants are never literals.
            name_prefix="__const_",
            update_var_mapping=True,
        )
        for sdfg_name, const_value in zip(sdfg_const_names, jaxpr.consts, strict=True):
            self._ctx.sdfg.add_constant(
                sdfg_name, copy.deepcopy(const_value), self._ctx.sdfg.arrays[sdfg_name]
            )

    def _allocate_translation_ctx(self, name: str | None = None) -> JaxprTranslationBuilder:
        """Allocate a new context and activate it.

        Args:
            name: The name of the SDFG.
        """
        self._ctx_stack.append(TranslationContext(name=name))
        return self

    @property
    def _ctx(self) -> TranslationContext:
        """Returns the currently active translation context."""
        assert len(self._ctx_stack) != 0, "No context is active."
        return self._ctx_stack[-1]

    def _clear_translation_ctx(self) -> TranslationContext | None:
        """Remove the currently active context from `self` and returns it.

        If `self` is not allocated it will return `None`.
        """
        if not self.is_allocated():
            return None

        if self.is_root_translator():
            # The translation, as a whole has finished, so restore the builder,
            #  i.e. delete all the shared state.
            self._jax_name_map = {}

        # Remove the current head stack.
        return self._ctx_stack.pop()

    def _translate_single_eqn(self, eqn: jax_core.JaxprEqn) -> None:
        """Translate `eqn` into its SDFG equivalent.

        To do this the function will perform the following steps:
        - Assemble the in and output variables.
        - Select the appropriate primitive translator to use.
        - Create a new empty state terminal state.
        - Call the primitive translator to perform the translation inside the new state.
        """
        if len(eqn.effects) != 0:
            raise NotImplementedError(f"Equation '{eqn}' has side effects.")

        # Input/Output variables
        #  Using a tuple for the input ensures that it cannot be modified.
        in_var_names: Sequence[str | None] = self.create_jax_var_list(
            eqn.invars,
            prevent_creation=True,  # Inputs must already exists.
            handle_literals=True,  #   but they can be literals.
        )
        out_var_names: Sequence[str] = self.create_jax_var_list(
            eqn.outvars,
            only_creation=True,  # Output must not exist yet.
            update_var_mapping=True,
        )

        primitive_name: str = eqn.primitive.name
        if primitive_name not in self._primitive_translators:
            raise NotImplementedError(f"No translator known to handle '{primitive_name}'.")
        translator = self._primitive_translators[primitive_name]

        # Create the state into which the equation should be translated
        eqn_state = self.append_new_state(
            label=f"{primitive_name}_{'_'.join(out_var_names)}",
            prev_state=None,  # forces the creation of a new terminal state
        )

        # Now perform the actual translation of the equation.
        new_sdfg_term_state = translator(
            builder=self,
            in_var_names=in_var_names,
            out_var_names=out_var_names,
            eqn=eqn,
            eqn_state=eqn_state,
        )

        # Determine the new (tentative) terminal state of the SDFG we are building.
        if new_sdfg_term_state is None:
            if eqn_state is not self._ctx.terminal_state:
                raise RuntimeError("Inconsistent terminal state was detected.")
            new_sdfg_term_state = eqn_state
        if not self._ctx.validate():
            raise RuntimeError("Detected an invalid SDFG under construction.")

        # Modify terminal root state of 'self'
        self._ctx.terminal_state = new_sdfg_term_state

    def _translate_jaxpr_internal(self, jaxpr: jax_core.ClosedJaxpr) -> TranslationContext:
        """Performs the actual translation of the Jaxpr into an SDFG.

        The function assumes that the context is allocated as well as the
        initial variables. The function removes and returns the currently
        active translation context.

        Args:
            jaxpr: The Jaxpr to translate.

        Notes:
            Equations that store into drop variables, i.e. with name `_`,
            will be ignored.
        """
        nb_translated_eqn: int = 0
        out_var_names: Sequence[str] = ()

        for eqn in jaxpr.jaxpr.eqns:
            if any(util.is_drop_var(outVar) for outVar in eqn.outvars):
                assert all(util.is_drop_var(outVar) for outVar in eqn.outvars)
                continue
            self._translate_single_eqn(eqn=eqn)
            nb_translated_eqn += 1

        # Handle the output or the case of an empty Jaxpr
        if nb_translated_eqn == 0:
            out_var_names = self._handle_null_jaxpr(jaxpr)
        else:
            out_var_names = self.create_jax_var_list(
                jaxpr.jaxpr.outvars, prevent_creation=True, handle_literals=False
            )

        self._ctx.out_names = tuple(out_var_names)

        return cast(TranslationContext, self._clear_translation_ctx())

    def _handle_null_jaxpr(self, jaxpr: jax_core.ClosedJaxpr) -> list[str]:
        """This function is called in case a `Jaxpr` with zero equations is encountered.

        A function with zero equation might still have output, in which case
        an input is copied to an output. This function will handle the copying
        from the input into the corresponding output variable. It is important
        that the function will remove the variables that are used as input and
        output from the mapping.

        Returns:
            The function returns a tuple containing the SDFG variables that
            refer to the output. The order of the list is the same as in
            `jaxpr.jaxpr.outvars`.

        Todo:
            - Handle the case if if the output is a literal.

        Note:
            The function will _not_ update the `out_names` field of the current context.
        """
        assert self._ctx.terminal_state is self._ctx.start_state
        assert isinstance(self._ctx.inp_names, tuple)
        assert self._ctx.out_names is None

        # There is not output so we do not have to copy anything around.
        if not jaxpr.out_avals:
            return []

        # List of the real output variables.
        out_var_names: list[str] = []

        # If we are here then we are dealing with a nested SDFG/Jaxpr, that has output.
        #  Because an input also serves as output, the nested SDFG will have a connector for the
        #  input and one for the output, but both with the same name. This will make node
        #  validation fail. We have to work around this by introducing some fake copies, which
        #  will be removed by DaCe later.
        for jax_out_var in jaxpr.jaxpr.outvars:
            # Since the output is also used as an input the variable mapping must be already known.
            sdfg_in_name: str = self.map_jax_var_to_sdfg(jax_out_var)

            # Now we create a variable that serves as true output, however, since the Jax variable
            #  is already known we can not update the variable mapping and must use another name.
            sdfg_out_name = self.add_array(
                jax_out_var, name_prefix="_zero_equation_output_for_", update_var_mapping=False
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

            # `jax_out_var` now has, in some sense, two SDFG equivalents, the input, that
            #  was previously created by `self._create_initial_input()` and the `sdfg_out_name`
            #  we just created. But we can not add this to the mapping. Because it is the best,
            #  as in the least worst thing we can do, we remove it from the mapping.
            #  I am open for different approaches.
            self._jax_name_map.pop(jax_out_var)

        return out_var_names

    @property
    def _start_state(self) -> dace.SDFGState:
        return cast(dace.SDFGState, self._ctx.start_state)

    @property
    def _terminal_sdfg_state(self) -> dace.SDFGState:
        """Returns the current terminal state of the SDFG under construction."""
        return cast(dace.SDFGState, self._ctx.terminal_state)


class TranslationContext:
    """Translation context used by the `JaxprTranslationBuilder`.

    Internal representation of the builder of an SDFG under construction together
    with the needed metadata. Essentially it is an extended version of the
    `TranslatedJaxprSDFG`, but carrying an unfinished canonical SDFG.
    A user should consider this class as an opaque object, that represents an
    invalid `TranslatedJaxprSDFG` object, and the only valid operation a user
    can do with it is passing it either to `finalize_translation_context()` or
    the `postprocess_jaxpr_sdfg()` function.

    Attributes:
        sdfg: The encapsulated SDFG object.
        inp_names: A list of the SDFG variables that are used as input
        out_names: A list of the SDFG variables that are used as output.
        start_state: The first state in the SDFG state machine.
        terminal_state: The (currently) last state in the state machine.

    Args:
        name: The name of the SDFG, will be forwarded to the encapsulated `TranslatedJaxprSDFG`.

    Note:
        Access of any attribute of this class by an outside user is considered undefined behaviour.
    """

    sdfg: dace.SDFG
    inp_names: tuple[str, ...] | None
    out_names: tuple[str, ...] | None
    start_state: dace.SDFGState
    terminal_state: dace.SDFGState

    def __init__(self, name: str | None = None) -> None:
        if isinstance(name, str) and not util.VALID_SDFG_OBJ_NAME.fullmatch(name):
            raise ValueError(f"'{name}' is not a valid SDFG name.")

        self.sdfg = dace.SDFG(name=(name or f"unnamed_SDFG_{id(self)}"))
        self.inp_names = None
        self.out_names = None
        self.start_state = self.sdfg.add_state(label="initial_state", is_start_block=True)
        self.terminal_state = self.start_state

    def validate(self) -> bool:
        """Validate internal state of `self`.

        Since the SDFG is under construction it will not be validated, instead the meta data
        will be validated.
        """
        if self.start_state is not self.sdfg.start_block:
            raise dace.sdfg.InvalidSDFGError(
                f"Expected to find '{self.start_state}' as start state,"
                f" but instead found '{self.sdfg.start_block}'.",
                self.sdfg,
                self.sdfg.node_id(self.start_state),
            )
        if {self.terminal_state} != set(self.sdfg.sink_nodes()):
            raise dace.sdfg.InvalidSDFGError(
                f"Expected to find as terminal state '{self.terminal_state}',"
                f" but instead found '{self.sdfg.sink_nodes()}'.",
                self.sdfg,
                self.sdfg.node_id(self.terminal_state),
            )
        if not (
            self.inp_names is None
            or all(inp_name in self.sdfg.arrays for inp_name in self.inp_names)
        ):
            raise dace.sdfg.InvalidSDFGError(
                f"Missing input arguments: {(inp_name for inp_name in self.inp_names if inp_name not in self.sdfg.arrays)}",
                self.sdfg,
                self.sdfg.node_id(self.terminal_state),
            )
        if not (
            self.out_names is None
            or all(out_name in self.sdfg.arrays for out_name in self.out_names)
        ):
            raise dace.sdfg.InvalidSDFGError(
                f"Missing output arguments: {(out_name for out_name in self.out_names if out_name not in self.sdfg.arrays)}",
                self.sdfg,
                self.sdfg.node_id(self.terminal_state),
            )
        return True
