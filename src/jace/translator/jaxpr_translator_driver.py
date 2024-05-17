# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, cast, overload

import dace
import jax
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
    Especially, DaCe's validation function will fail and it is unable to be perocessed by the optimization pipeline.
    For more information also see `jace.translator.post_translation` module for more information.

    The idea of the translator is extremely simple. Since Jaxpr is a list
    consisting of more or less simple instructions/equations, they get processed
    one after the other. Each equation is translated into its own state that
    is appended to the SDFG, thus the SDFG is a long list of states. In certain
    cases it might be that an equation needs more states, but this is an exception.

    The actual translation of the equation is not handled by the driver.
    Instead the request is forwarded to a `PrimitiveTranslator` object, also known as subtranslator.
    This is a highly specialized object that is able to handle one kind of primitive.
    For more information on the subtranslators see the documentation of `PrimitiveTranslator`.
    The actual translators are supplied from the outside at construction time.

    To start a translation the `translate_jaxpr()` function should be called,
    if this happens it is said that the driver has an ongoing translation.
    If `translate_jaxpr()` is called on driver that has an ongoing translation, a new translation context will be set up.
    Thus the driver will then translate the supplied (nested) Jaxpr and return the result.
    However, this will have no influence on the translation process that is already going.

    Notes:
        The translator is able to handle multiple consecutive translations.
    """

    __slots__ = (
        "_ctx_stack",  # Stack of all contexts
        "_reserved_names",  # Part of the context, but is copied.
        "_sub_translators",
        "_rev_manager",
    )

    def __init__(
        self,
        sub_translators: Mapping[str, translator.PrimitiveTranslator],
    ) -> None:
        """Creates the driver.

        Args:
            sub_translators:    Use these subtranslators to perform the translation.

        Notes:
            `sub_translators` is not copied, thus the user has to guarantee,
                that it will not change during translation.
                It is highly advised but not required to use the output of
                `get_subtranslators()` or pass a copy as argument.
        """

        # Shared with the outside, while key and mapped values are immutable,
        #  the mapping itself is not, but it should be fine.
        #  Allocated through the lifetime of `self`.
        self._sub_translators: Mapping[str, translator.PrimitiveTranslator] = sub_translators

        # These names can not be used for the automatic naming of Jax variables.
        #  They differ from the forbidden names, that they denote valid SDFG names.
        #  An example would be names of the function arguments.
        #  Only allocated during an ongoing translation.
        self._reserved_names: set[str] = None  # type: ignore[assignment]

        # Shared revision counter manager.
        #  Generates the revision numbers we need.
        #  Is reset after every translation.
        self._rev_manager: itertools.count[int] = itertools.count(0, 1)

        # Context stack and current context.
        #  Only allocated during an ongoing translation
        self._ctx_stack: list[translator.TranslatedJaxprSDFG] = []

    def translate_jaxpr(
        self,
        jaxpr: jax_core.ClosedJaxpr,
        *,
        inp_scalar_as_array: bool = False,
        name: str | None = None,
        reserved_names: str | Iterable[str] = (),
    ) -> translator.TranslatedJaxprSDFG:
        """Perform the translation of a Jaxpr into a SDFG.

        In case this function is called and `self` has an ongoing translation process, a new translation context will be created.
        This means the Jaxpr will be translated independently from the previous one.

        Returns:
            The function will translate the passed Jaxpr object into an SDFG in canonical form.
            This SDFG together with additional meta data, that is needed for further processing
            is encapsulated inside a `TranslatedJaxprSDFG` object.

        Args:
            inp_scalar_as_array:    Translate scalar _input_ arguments to arrays of length 1.
            name:                   Use this name for the SDFG instead some generated one.
            reserved_names:         Prevent the generation of variables with these names, see `self.add_array()` for more.
        """
        if len(jaxpr.effects) != 0:
            raise NotImplementedError("'Jaxpr' with side effects are not supported.")
        if not jax.config.read("jax_enable_x64"):
            raise NotImplementedError("The translation only works if 'jax_enable_x64' is enabled.")

        # NOTE: If `self` is already allocated, i.e. has an ongoing translation process,
        #       the `_allocate_translation_ctx()` function will start a new context.
        #       Thus the driver will start to translate a second (nested) SDFG.
        #       Also note that there is no mechanism that forces the integration of the nested SDFG/Jaxpr,
        #       this must be done manually.
        self._allocate_translation_ctx(
            name=name,
            reserved_names=reserved_names,
        )
        self._create_constants(
            jaxpr=jaxpr,
        )
        self._create_initial_input(
            jaxpr=jaxpr,
            inp_scalar_as_array=inp_scalar_as_array,
        )
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

        By default the new state is appended to the current terminal state.
        This will also update the terminal SDFG state of `self`.

        However, if `prev_state` is specified the state new state will be
        appended to `prev_state` instead. This will not modify the terminal
        state unless `prev_state` is the current terminal state.

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
        """Get all `Data` descriptors that are currently known to the SDFG.

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
        In case it is a `jax.core.Atom` it is first translated, see `self.map_jax_var_to_sdfg()`.
        """
        if isinstance(name, str):
            sdfg_name: str = name
        elif isinstance(name, (jax_core.Var, util.JaCeVar)):
            sdfg_name = self.map_jax_var_to_sdfg(name)
        else:
            raise TypeError(f"Does not know how to handle '{type(name).__name__}'.")
        if sdfg_name not in self._ctx.sdfg.arrays:
            raise KeyError(f"Requested SDFG array '{name}' but it is not known.")
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
            allow_fail:     If mapping is not known return `None` instead of raise `KeyError`.
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

    @property
    def terminal_sdfg_state(self) -> dace.SDFGState:
        """Returns the current terminal state of the SDFG under construction.

        The SDFGs that are constructed by the driver are essentially a list of states.
        New states are appended at the current terminal/end state and becoming the new terminal state.
        This function returns the current terminal state.
        """
        return cast(dace.SDFGState, self._ctx.terminal_state)

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
        if self._ctx.rev_idx == 0:
            return True
        return False

    @property
    def rev_idx(self) -> int:
        """Returns the revision index of `self`."""
        if not self.is_allocated():
            raise RuntimeError("Driver is not allocated.")
        return cast(int, self._ctx.rev_idx)

    def add_jax_name_mapping(
        self,
        jax_var: jax_core.Var | util.JaCeVar,
        sdfg_name: str,
    ) -> JaxprTranslationDriver:
        """Creates a mapping between `jax_var` to `sdfg_name`.

        This function updates the internal map of `self` and after the call
        `self.map_jax_var_to_sdfg()` will identify `jax_var` with `sdfg_name`.
        This function is not able to delete a variable mapping that was
        established before, for this use TBA.

        Args:
            jax_var:     The Jax variable.
            sdfg_name:   The name of the corresponding SDFG variable.
        """
        assert len(sdfg_name) > 0

        if jax_var in self._jax_name_map:
            if self._jax_name_map[jax_var] == sdfg_name:  # noops.
                return self
            raise ValueError(
                f"Tried to create the mapping '{jax_var} -> {sdfg_name}', but '{jax_var}'"
                f" already points to '{self.map_jax_var_to_sdfg(jax_var)}'."
            )
        if sdfg_name not in self._ctx.sdfg.arrays:
            raise KeyError(f"Mapping '{jax_var} -> {sdfg_name}': SDFG target unknown.")
        if sdfg_name in self._forbidden_names:
            raise NameError(f"Mapping '{jax_var} -> {sdfg_name}': Forbidden name.")

        self._jax_name_map[jax_var] = sdfg_name
        return self

    def add_reserved_names(
        self,
        reserved_names: str | Iterable[str],
    ) -> JaxprTranslationDriver:
        """Adds the names listed in `reserved_names` to the internal list."""

        if not reserved_names:
            return self
        if isinstance(reserved_names, str):
            reserved_names = [reserved_names]
        self._reserved_names.update(reserved_names)
        return self

    def add_array(
        self,
        arg: jax_core.Atom | util.JaCeVar,
        *,
        as_transient: bool = True,
        alt_name: str | None = None,
        name_prefix: str | None = None,
        find_new_name: bool | None = None,
        force_array: bool = False,
        strides: Sequence[int | dace.symbol | str] | None = None,
        allow_literals: bool = False,
        force_jax_name: bool = False,
        update_var_mapping: bool = False,
    ) -> str:
        """Creates an SDFG variable for the Jax variable `arg` and returns its SDFG name.

        By default the function will create a transient, use `as_transient=True` to change that.
        By default the function will honor if the Jax variable is a scalar or an array.
        However, by setting `force_array` the function will always generate an array.

        By default the name for the SDFG variable is derived from the Jax variable.
        It is guaranteed that this name is unique in the SDFG, even in the presence of nested SDFGs.
        By specifying `alt_name` it is possible to force a certain name on a variable.
        It is important that if `alt_name` is specified the function will either generate the variable or fail.

        The  driver distinguishes between two kinds of "bad (SDFG) variable names".
        The first category are the forbidden names, which the function refuses to generate.
        The second type are the so called reserved names, which were set at the beginning, or by `self.add_reserved_names()`.
        These names can be used if the name is specified through `alt_name` but are not used in automatic naming.

        If nothing is specified, the strides of the data are determined by DaCe, which is continuous C order.
        It is possible to set a certain values by setting `strides` appropriate.

        By default this function does not update the internal variable map.
        However, by setting `update_var_mapping` to `True` the function will
        update the mapping.

        Args:
            arg:                The Jax object for which a SDFG equivalent should be created.
            as_transient:       If set, the SDFG variable is a transient, `True` by default.
            alt_name:           Try to create the variable with this name; either succeed or fail.
            name_prefix:        If given and in automatic naming mode, add this prefix to the name.
            find_new_name:      The translator will try to find a new name if the designated is already occupied.
            force_array:        Instead of a `dace.Scalar` create a `dace.Array` with one element.
            strides:            Instead of the default strides use these values.
            allow_literals:     If `True` then also allows JaxLiterals as `arg`.
            force_jax_name:     If `True` then, the verbatim Jax name will be used.
            update_var_mapping: Update the internal variable mapping; by default `False`.

        Notes:
            If this function is used directly a user is advised to always set
                `update_var_mapping` to `True`.
            If you need to create a special array, you can use `jace.util.JaCeVar`
                to create a pseudo Jax variable.
        """
        shape: tuple[int] = util.get_jax_var_shape(arg)
        dtype = util.get_jax_var_dtype(arg)
        offset = None  # i.e. no offset
        storage: dace.StorageType = dace.StorageType.Default  # Set at later stages (optimization)
        is_scalar: bool = shape == ()

        if force_jax_name:
            if alt_name is not None:
                raise ValueError(
                    f"Specified 'force_jax_name', but passed '{alt_name}' as 'alt_name'."
                )
            if name_prefix is not None:
                raise ValueError(
                    f"Specified 'force_jax_name', but passed '{name_prefix}' as 'name_prefix'."
                )
            if find_new_name:
                raise ValueError("Specified `force_jax_name` but also wanted a new name.")
            find_new_name = False
            alt_name = util.propose_jax_name(arg, self._jax_name_map)
        if alt_name is not None:
            find_new_name = False  # If a name was given, then use it no matter what.
            if len(alt_name) == 0:
                raise ValueError("Passed an empty 'alt_name'.")
            if alt_name in self._forbidden_names:
                raise ValueError("'alt_name' is a forbidden name.")
            if not util.VALID_SDFG_VAR_NAME.fullmatch(alt_name):
                raise ValueError(f"The passed name 'alt_name' '{alt_name}' is invalid.")
            if update_var_mapping and arg in self._jax_name_map:
                raise ValueError(f"Variable '{alt_name}' already registered.")
            if alt_name in self._ctx.sdfg.arrays:
                raise ValueError(f"Variable '{alt_name}' already exists.")
            if name_prefix is not None:
                raise ValueError(
                    f"Specified 'name_prefix' ('{name_prefix}') but passed '{alt_name}' as 'alt_name'."
                )
        if (name_prefix is not None) and (len(name_prefix) == 0):
            raise ValueError("Specified an empty 'name_prefix'.")

        # Now we determine the proposed name of the variable.
        #  Depending on the situation, we will further manipulate it.
        if alt_name is not None:
            prop_name = alt_name  # Just for completion: will be ignored later
        elif isinstance(arg, (jax_core.Var, util.JaCeVar)):
            prop_name = util.propose_jax_name(arg, self._jax_name_map)
            assert not prop_name.startswith("__")
            if name_prefix is not None:
                prop_name = name_prefix + prop_name
        elif isinstance(arg, jax_core.Literal):  # type: ignore[unreachable]
            if not allow_literals:  # Allows to use a literal as template.
                raise NotImplementedError("Jax Literals are not supported.")
            if alt_name is None:
                raise ValueError(f"Passed literal '{arg}', but not specified a name to use.")

        if alt_name is None:
            # If we are the root translator, then we will use `prop_name` directly;
            #  otherwise we will append the revision of `self` to the name.
            arg_name = prop_name + (
                "" if self.is_root_translator() else f"_rev_idx{self._ctx.rev_idx}"
            )
        else:
            # Use the supplied name directly.
            arg_name = str(alt_name)

        # Checking the strides.
        if strides is not None:
            if is_scalar:
                raise ValueError("Specified a stride for a scalar.")
            if isinstance(strides, (str, dace.symbol, int)):
                strides = (strides,)
            elif not isinstance(strides, tuple):
                strides = tuple(strides)
            if len(strides) != len(shape):
                raise ValueError(
                    f"'strides' has length {len(strides)}, but array rank is {len(shape)}."
                )

        # Determine if we should look for a new name or not, if nothing was specified
        if find_new_name is None:
            if arg_name in self._reserved_names:
                find_new_name = True
            if arg_name in self._forbidden_names:
                # This is not an error, but happens if we handle Jax variable `if`.
                find_new_name = True

        if find_new_name:
            name_tmpl = "_jax_variable__" + arg_name + "__{}"
            for iCounter in range(1000):
                _arg_name = name_tmpl.format(iCounter)
                if (
                    (_arg_name in self._forbidden_names)
                    or (_arg_name in self._reserved_names)
                    or (_arg_name in self._ctx.sdfg.arrays)
                ):
                    continue  # The proposed variable is known, so try next value.
                arg_name = _arg_name  # We found a name that we can use.
                break
            else:
                raise ValueError(f"Failed to find a replacement name for '{arg_name}'")
            del iCounter, _arg_name

        # Final name check
        if arg_name in self._forbidden_names:
            raise ValueError(f"Can't create variable '{arg_name}', name is forbidden.")
        if arg_name in self._ctx.sdfg.arrays:
            raise ValueError(f"Can't create variable '{arg_name}', variable is already created.")
        if not util.VALID_SDFG_VAR_NAME.fullmatch(arg_name):
            raise ValueError(f"The requested variable name '{arg_name}' is invalid.")

        # Promotion of scalar to array.
        if is_scalar and force_array:
            shape = (1,)
            strides = None
            is_scalar = False

        if is_scalar:
            self._ctx.sdfg.add_scalar(
                name=arg_name, storage=storage, dtype=dtype, transient=as_transient
            )
        else:
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
            self.add_jax_name_mapping(jax_var=arg, sdfg_name=arg_name)

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
        If no SDFG variable is known the function will create one using `add_array()`, with `update_var_mapping` set to `True`.

        By setting `prevent_creation` the function will not create any new SDFG variables.
        This mode is used to indicate that all variables have to exists already.
        By setting `only_creation` the function will only create new SDFG variables.
        If a Jax variable already has a known SDFG equivalent an error is generated.

        By default literals cause an error.
        However, by setting `handle_literals` to `True` literals will will be included in the output with the value `None`.

        Args:
            jax_var_list:       The list of Jax variables that should be transformed to SDFG names.
            prevent_creation:   Never create a variable, all must already be known.
            only_creation:      Always create a variable, it is an error if one already exist.
            handle_literals:    Allow the processing of literals.
            kwargs:             Will be forwarded to `self.add_array()` if a variable as to be created,

        Todo:
            Rollback if the creation fails.
        """
        if only_creation and prevent_creation:
            raise ValueError("Specified both 'only_creation' and 'prevent_creation'.")
        assert (
            "update_var_mapping" not in kwargs
        ), "You can not pass 'update_var_mapping' as argument to 'create_jax_var_list()'."

        ret_list: list[None | str] = []
        for jax_var in jax_var_list:
            if isinstance(jax_var, jax_core.Literal):
                if not handle_literals:
                    raise ValueError("Encountered a literal but `handle_literals` was `False`.")
                sdfg_name = None
            else:
                mapped_sdfg_name: str | None = self.map_jax_var_to_sdfg(jax_var, allow_fail=True)
                if (mapped_sdfg_name is None) and prevent_creation:
                    raise ValueError(f"'prevent_creation' given but have to create '{jax_var}'.")
                if mapped_sdfg_name is None:
                    sdfg_name = self.add_array(arg=jax_var, update_var_mapping=True, **kwargs)
                elif only_creation:
                    raise ValueError(f"'only_creation' given '{jax_var}' already exists.")
                else:
                    sdfg_name = mapped_sdfg_name
                # Calling `add_jax_name_mapping` is save, because if the mapping does already exists it is a no ops.
                self.add_jax_name_mapping(jax_var, sdfg_name)

            ret_list.append(sdfg_name)

        return ret_list

    def _create_initial_input(
        self,
        jaxpr: jax_core.ClosedJaxpr,
        inp_scalar_as_array: bool,
    ) -> Sequence[str]:
        """This function will create the internal input variables that are used for the SDFG.

        Args:
            jaxpr:                  The Jaxpr that we want to translate.
            inp_scalar_as_array:    Promote scalars to arrays of size one.

        Returns:
            The list of SDFG variables used as input arguments of `jaxpr` in the same order.

        Notes:
            This function will fill the internal list of inputs.
        """
        if not self.is_allocated():
            raise RuntimeError("Driver is not allocated, can not create constants.")
        if len(self._ctx.inp_names) != 0:
            raise RuntimeError("Called '_create_initial_input()' twice?")

        # Handle the initial input arguments
        sdfg: dace.SDFG = self._ctx.sdfg
        init_in_var_names: Sequence[str] = self.create_jax_var_list(
            jax_var_list=jaxpr.jaxpr.invars,
            only_creation=True,
            as_transient=True,  # Explicit transient; no error!
            handle_literals=False,  # Initial arguments are never literals
            force_array=inp_scalar_as_array,
            force_jax_name=self.is_root_translator(),  # Ensure root get pure Jax names.
        )
        # This forces the code to only accept kwargs
        #  Is also part of "what a canonical sdfg" is.
        sdfg.arg_names = []

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
            only_creation=True,
            strides=None,
            name_prefix="__const_",
        )

        for sdfg_name, const_value in zip(sdfg_const_names, jaxpr.consts, strict=True):
            # We have to pass the data descriptor to `add_constant()`, otherwise a new one would be created.
            self._ctx.sdfg.add_constant(
                sdfg_name, deepcopy(const_value), self._ctx.sdfg.arrays[sdfg_name]
            )
        return sdfg_const_names

    def _allocate_translation_ctx(
        self,
        name: str | None = None,
        reserved_names: str | Iterable[str] = (),
    ) -> JaxprTranslationDriver:
        """This function allocates and initialize the members of the translation context of `self`.

        If this function is called and `self` is already allocated, the function will create a new context.
        This allows the driver to handle nested Jaxpr.
        The first context that is created is also known as root translator.

        Args:
            name:               The name of the SDFG.
            reserved_names:     Add these name to the set of resered names of `self`.
        """
        from jace import translator  # Cyclic import

        # Create a new translation context and put it on the stack.
        self._ctx_stack.append(
            translator.TranslatedJaxprSDFG(
                rev_idx=next(self._rev_manager),
                name=name,
            )
        )

        if self.is_root_translator():
            # The root translation, i.e. the very first context allocation
            #  Thus we also have to allocate the additional members
            #  which are shared among all contexts.
            self._reserved_names = set()
            self.add_reserved_names(reserved_names)

        else:
            # We are in a nested context.
            #  We might have to update the reserved names.
            self.add_reserved_names(reserved_names)

        return self

    @property
    def _ctx(self) -> translator.TranslatedJaxprSDFG:
        """Returns the currently active translation context."""
        assert len(self._ctx_stack) != 0, "No context is active."
        return self._ctx_stack[-1]

    def _clear_translation_ctx(self) -> JaxprTranslationDriver:
        """This function deallocate the translation context of `self`.

        Notes:
            While it is allowed for outside code to call this function explicit
                it is is most likely an error.
            If `self` is not allocated this function acts as a noops.
            The reserved names are only deallocated if `self` is a root translator.
        """
        if not self.is_allocated():
            return self

        if self.is_root_translator():
            self._rev_manager = itertools.count(0, 1)
            self._reserved_names = None  # type: ignore[assignment]
            self._ctx_stack.pop()

        else:
            # Restore the previous state
            self._ctx_stack.pop()
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
                handle_literals=True,  #  but they can be literals.
            )
        )
        out_var_names: MutableSequence[str] = self.create_jax_var_list(
            eqn.outvars,
            only_creation=True,  # Output must not exist yet.
        )

        # Find the subtranslator
        prim_name: str = eqn.primitive.name
        if prim_name not in self._sub_translators:
            raise NotImplementedError(f"No subtranslators known to handle '{prim_name}'.")
        subtranslator = self._sub_translators[prim_name]

        # Create the state into which the equation should be translated
        last_term_state: dace.SDFGState = self.terminal_sdfg_state  # noqa: F841 # Will be used later
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
        #  but he must update the `out_var_names` list correctly, we will now verify this.
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

        The function assumes that the context is allocated as well as initial variables.
        The function will return the internal state of `self` as a `TranslatedJaxprSDFG` object.
        However, it will not deallocate the translation context, thus `self` and the return value share the same memory.

        Args:
            jaxpr:      The Jaxpr to translate.

        Notes:
            The function will unconditionally handle empty Jaxpr.
            Jax uses a variable with name `_` to indicate that this value is never read,
                this is used by Jax to indicate that they are never read.
                Such variables are included by some transformations such as `grad()`.
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
        if len(jaxpr.eqns) != 0:
            raise NotImplementedError("'_handle_null_jaxpr()' was called for a non empty Jaxpr.")
        if len(jaxpr.out_avals) == 0:
            # There is not output so we do not have to copy anything around.
            return ()
        assert self._ctx.terminal_state is self._ctx.start_state
        assert len(self._ctx.inp_names) > 0
        assert len(self._ctx.out_names) == 0

        # List of the output variables.
        out_var_names: list[str] = []

        # If we are here then we are dealing with a nested SDFG/Jaxpr.
        #  Because an input also serves as output, the nested SDFG will have a connector for the
        #  input and one for the output, but both with the same name.
        #  This will make node validation fail.
        #  We have to work around by introducing some fake copies, which will be removed by DaCe later.
        for jax_out_var in jaxpr.jaxpr.outvars:
            # Since the output is also used as an input the variable mapping must be known.
            sdfg_in_name: str = self.map_jax_var_to_sdfg(jax_out_var)

            # Now we create a variable that serves as true output, however, since the Jax variable
            #  is already known we can not update the variable mapping.
            sdfg_out_name = self.add_array(
                jax_out_var,
                as_transient=True,
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

            # A Jax variable now has two SDFG equivalent, the input, that was previously created by
            #  `self._create_initial_input()` and the `sdfg_out_name` we just created.
            #  But we can not add this to the mapping, because of this situation we will now remove
            #  the variable from the mapping. I am open for different approaches.
            #  Note that input variables that are not used, will remain in the mapping.
            self._jax_name_map.pop(jax_out_var)

        return tuple(out_var_names)

    @property
    def _jax_name_map(self) -> dict[jax_core.Var | util.JaCeVar, str]:
        return cast(dict[jax_core.Var | util.JaCeVar, str], self._ctx.jax_name_map)

    @property
    def _start_state(self) -> dace.SDFGState:
        return cast(dace.SDFGState, self._ctx.start_state)

    # fmt: off
    _forbidden_names: Final[set[str]] = {
        # These should be most of the C++ keywords, it is more important to have the short ones.
        #  Taken from 'https://learn.microsoft.com/en-us/cpp/cpp/keywords-cpp?view=msvc-170'
        'alignas', 'alignof', 'and', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
        'catch', 'char', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr',
        'constinit', 'continue', 'decltype', 'default', 'delete', 'directive', 'do', 'double',
        'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend',
        'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not',
        'nullptr', 'operator', 'or', 'private', 'protected', 'public', 'register', 'requires',
        'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'template', 'this',
        'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
        'virtual', 'void', 'volatile', 'while', 'xor', 'std',
    }
    # fmt: on
