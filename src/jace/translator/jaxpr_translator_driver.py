# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Final, cast, overload

import dace
import jax
from dace import data as ddata, properties as dprop
from jax import core as jcore

from jace import translator as jtrans, util as jutil


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
    TBA where to look for them.

    The idea of the translator is extremely simple. Since Jaxpr is a list
    consisting of more or less simple instructions/equations, they get processed
    one after the other. Each equation is translated into its own state that
    is appended to the SDFG, thus the SDFG is a long list of states. In certain
    cases it might be that an equation needs more states, but this is an exception.

    The actual translation of the equation is not handled by the driver.
    Instead the request is forwarded to a `PrimitiveTranslator` object, also known as subtranslator.
    This is a highly specialized object that is able to handle one kind of primitive.
    For more information on the subtranslators see the documentation of `PrimitiveTranslator`.

    To start a translation the `translate_jaxpr()` function should be called,
    if this happens it is said that the driver has an ongoing translation.
    If `translate_jaxpr()` is called on driver that has an ongoing translation, a new translation context will be set up.
    Thus the driver will then translate the supplied (nested) Jaxpr and return the result.
    However, this will have no influence on the translation process that is already going.
    """

    __slots__ = (
        "_ctx_stack",  # Stack of all contexts
        "_ctx",  # Current top of the context stack.
        "_reserved_names",  # Part of the context, but is copied.
        "_sub_translators",
        "_rev_manager",
    )

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Creates the base translator.

        All arguments that does not start with an underscore are used as
        arguments to construct the subtranslators.

        Notes:
            This function will not allocate the translation context of `self`
                but will only allocate the shared members.
            By setting `_no_shared_alloc` to `True` the function will not allocate
                the shared part. This flag is provided only for implementing
                `self.fork()` using it is an error and undefined behaviour.
        """
        from ._translation_context import _TranslationContext

        # Contains all the subtranslators that we need.
        #  They are partitioned by the names of the primitive they have registered for.
        #  This member is allocated by '_init_sub_translators()' and remains allocated
        #  during the lifetime of the object.
        self._sub_translators: dict[str, jtrans.PrimitiveTranslator] = None  # type: ignore[assignment]

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
        self._ctx_stack: list[_TranslationContext] = []
        self._ctx: _TranslationContext = None  # type: ignore[assignment]

        # Creating of the subtranslators.
        self._init_sub_translators(kwargs)

    def translate_jaxpr(
        self,
        jaxpr: jcore.ClosedJaxpr,
        *,
        inp_scalar_as_array: bool = False,
        name: str | None = None,
        reserved_names: str | Collection[str] | None = None,
        allow_empty_jaxpr: bool = False,
        **kwargs: Any,
    ) -> jtrans.TranslatedJaxprSDFG:
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
            reserved_names:         Prevent the generation of variables with these names,
                                        see `self.add_array()` for more.
            allow_empty_jaxpr:      Allows empty Jaxpr.

        Notes:
            Every time this function is called a new revision index is generated.
        """
        if (len(jaxpr.eqns) == 0) and (not allow_empty_jaxpr):
            raise ValueError("Passed an empty Jaxpr, but did not allow for empty Jaxpr.")
        if not isinstance(jaxpr, jcore.ClosedJaxpr):
            raise TypeError(f"Expected a 'jax.core.ClosedJaxp' instance but got '{type(jaxpr)}'")
        if len(jaxpr.effects) != 0:
            raise NotImplementedError("'Jaxpr' with side effects are not supported.")
        if len(jaxpr.out_avals) == 0:
            raise ValueError("Jaxpr has zero output variables.")
        if not jax.config.read("jax_enable_x64"):
            raise NotImplementedError("The translation only works if 'jax_enable_x64' is enabled.")

        # Consume the hidden flags
        _clear_translation_ctx: bool = kwargs.pop("_clear_translation_ctx", True)

        # NOTE: If `self` is already allocated, i.e. has an ongoing translation process
        #       This function will create a new translation context. Thus the driver
        #       will start to translate a second (nested) SDFG.
        #       Also note that there is no mechanism that forces the integration of the
        #       nested SDFG/Jaxpr.
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
        jsdfg: jtrans.TranslatedJaxprSDFG = self._translate_jaxpr_internal(jaxpr)

        # If the translation context is not cleared `self` and `jsdfg` will share the same data.
        #  There is some legitimate use for that.
        if _clear_translation_ctx:
            self._clear_translation_ctx()

        return jsdfg

    def append_new_state(
        self,
        label: str | None = None,
        condition: dprop.CodeBlock | None = None,
        assignments: Mapping[str, Any] | None = None,
        *,
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

    def get_arrays(self) -> Mapping[str, ddata.Data]:
        """Get all `Data` descriptors that are currently known to the SDFG.

        Notes:
            Essentially a shorthand and preferred way for `self.get_sdfg().arrays`.
            For getting a specific data descriptor use `self.get_array()`.
        """
        return cast(Mapping[str, ddata.Data], self._ctx.sdfg.arrays)

    def get_array(
        self,
        name: str | jcore.Atom | jutil.JaCeVar,
    ) -> ddata.Data:
        """Returns the SDFG `Data` object `name` referees to.

        If `name` is a string it is directly interpreted as the name of an SDFG variable.
        In case it is a `jax.core.Atom` it is first translated, see `self.map_jax_var_to_sdfg()`.
        """
        if isinstance(name, str):
            sdfg_name: str = name
        elif isinstance(name, (jcore.Var, jutil.JaCeVar)):
            sdfg_name = self.map_jax_var_to_sdfg(name)
        else:
            raise TypeError(f"Does not know how to handle '{type(name).__name__}'.")
        if sdfg_name not in self._ctx.sdfg.arrays:
            raise KeyError(f"Requested SDFG array '{name}' but it is not known.")
        return self._ctx.sdfg.arrays[sdfg_name]

    @overload
    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jcore.Atom | jutil.JaCeVar,
    ) -> str: ...

    @overload
    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jcore.Atom | jutil.JaCeVar,
        allow_fail: bool,
    ) -> str | None: ...

    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jcore.Atom | jutil.JaCeVar,
        allow_fail: bool = False,
    ) -> str | None:
        """Get the _name_ of the SDFG variable to which `jax_var` is referring to.

        For convenient this function will consider a string as input to be already an SDFG variable name.

        Args:
            jax_var:        The Jax variable to look up.
            allow_fail:     If mapping is not known return `None` instead of raise `KeyError`.
        """
        if isinstance(jax_var, str):
            sdfg_name: str = jax_var
        elif isinstance(jax_var, jcore.Literal):
            raise RuntimeError("There is no SDFG variable for literal '{jax_var}'.")
        elif jax_var in self._ctx.jax_name_map:
            sdfg_name = self._ctx.jax_name_map[jax_var]
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

    def get_sdfg(self) -> dace.SDFG:
        """Returns the SDFG that is currently constructed.

        If you want access to the arrays of the SDFG use `self.get_arrays()`/`self.get_array()`.
        """
        return self._ctx.sdfg

    def get_terminal_sdfg_state(self) -> dace.SDFGState:
        """Returns the current terminal state of the SDFG under construction.

        The SDFGs that are constructed by the driver are essentially a list of states.
        New states are appended at the current terminal/end state and becoming the new terminal state.
        This function returns the current terminal state.
        """
        return self._ctx.terminal_state

    def is_allocated(self) -> bool:
        """Tests if `self` has an allocated context.

        If `self` is allocated then there is also an ongoing translation process.
        """
        assert isinstance(self._sub_translators, dict)
        if self._ctx is not None:
            assert self._ctx_stack[-1] is self._ctx
            return True
        assert len(self._ctx_stack) == 0  # type: ignore[unreachable]
        return False

    def is_root_translator(self) -> bool:
        """Tests if `self` is a root translator.

        The root translator (context) is the very first translator process that was started.
        """
        if not self.is_allocated():
            raise RuntimeError("Driver is not allocated.")
        if self._ctx.rev_idx == 0:
            assert len(self._ctx_stack) == 1
            return True
        return False

    def get_rev_idx(self) -> int:
        """Returns the revision index of `self`."""
        if not self.is_allocated():
            raise RuntimeError("Driver is not allocated.")
        return self._ctx.rev_idx

    def add_jax_name_mapping(
        self,
        jax_var: jcore.Var | jutil.JaCeVar,
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
        assert isinstance(sdfg_name, str) and (len(sdfg_name) > 0)  # noqa: PT018  # Should be one assertion.

        if jax_var in self._ctx.jax_name_map:
            if self._ctx.jax_name_map[jax_var] == sdfg_name:  # noops.
                return self
            raise ValueError(
                f"Tried to create the mapping '{jax_var} -> {sdfg_name}', but '{jax_var}'"
                f" already points to '{self.map_jax_var_to_sdfg(jax_var)}'."
            )
        if sdfg_name not in self.get_arrays():
            raise KeyError(f"Mapping '{jax_var} -> {sdfg_name}': SDFG target unknown.")
        if sdfg_name in self._forbidden_names:
            raise NameError(f"Mapping '{jax_var} -> {sdfg_name}': Forbidden name.")

        self._ctx.jax_name_map[jax_var] = sdfg_name
        return self

    def add_reserved_names(
        self,
        reserved_names: None | str | Collection[str],
    ) -> JaxprTranslationDriver:
        """Adds the names listed in `reserved_names` to the internal list."""
        assert isinstance(self._reserved_names, set)

        if reserved_names is None:
            return self
        if isinstance(reserved_names, str):
            reserved_names = [reserved_names]
        elif isinstance(reserved_names, Collection):
            pass
        else:
            raise TypeError(f"Does not know how to handle the type '{type(reserved_names)}'.")
        if not all(isinstance(x, str) and (len(x) != 0) for x in reserved_names):
            raise TypeError("Reserved names must all be non empty strings.")
        self._reserved_names.update(reserved_names)
        return self

    def add_array(
        self,
        arg: jcore.Atom | jutil.JaCeVar,
        *,
        as_transient: bool = True,
        alt_name: str | None = None,
        name_prefix: str | None = None,
        force_array: bool = False,
        as_view: bool = False,
        strides: Sequence[int | dace.symbol | str] | None = None,
        symb_strides: bool | None = None,
        find_new_name: bool | None = None,
        allow_literals: bool = False,
        force_jax_name: bool = False,
        update_var_mapping: bool = False,
    ) -> str:
        """Creates an SDFG variable for the Jax variable `arg` and returns its SDFG name.

        By default the function will create a transient, use `as_transient` to
        change that. By default the function will honor if the Jax variable is
        a scalar or an array. However, by setting `force_array` the function
        will always generate an array.

        By default the name for the SDFG variable is derived from the Jax variable.
        It is guaranteed that this name is unique in the SDFG, even in the presence
        of nested SDFGs. By specifying `alt_name` it is possible to force a certain
        name on a variable. It is important that if `alt_name` is specified the function
        will either generate the variable or fail.

        The  driver distinguishes between two kinds of "bad (SDFG) variable names".
        The first category are the forbidden names, which the function refuses to generate.
        The second type are the so called reserved names, which were set at the beginning.
        These names can be used if they are specified through `alt_name` but are not used
        in automatic naming.

        If nothing is specified, the strides of the data are determined by DaCe, which is
        continuous C order. There are two ways to change that.
        The first way is to specify the `strides` argument, which are then forwarded
        to the underlying DaCe function. The second way is to set `symb_strides`
        to `True` in which case the function will generate symbols and use them.
        However, even if symbolic strides are activated, arrays with just one
        dimensions have always a non symbolic stride of 1. Furthermore, dimensions
        with shape 1 will always have stride 0.

        By default this function does not update the internal variable map.
        However, by setting `update_var_mapping` to `True` the function will
        update the mapping.

        Args:
            arg:                The Jax object for which a SDFG equivalent should be created.
            as_transient:       If set, the SDFG variable is a transient, `True` by default.
            alt_name:           Try to create the variable with this name; either succeed or fail.
            name_prefix:        If given and in automatic naming mode, add this prefix to the name.
            force_array:        Instead of a `dace.Scalar` create a `dace.Array` with one element.
            as_view:            Creates a view instead of an array, if it is a scalar
                                    it is silently ignored.
            strides:            Instead of the default strides use these values.
            symb_strides:       Create symbols and use them for fully symbolic strides.
            find_new_name:      The translator will try to find a new name if the designated
                                    is already occupied. This does not work if the name
                                    was supplied by `alt_name`.
            allow_literals:     If `True` then also allows JaxLiterals as `arg`.
            force_jax_name:     If `True` then, the verbatim Jax name will be used.
            update_var_mapping: Update the internal variable mapping; by default `False`.

        Notes:
            If this function is used directly a user is advised to always set
                `update_var_mapping` to `True`.
            If `find_new_name` is `None` the default, the function will only
                look for a new name if there is a need for it. If it is `True`
                the function will always look for a new name, even if the initial
                name was fine. If it is `False` the function will never look for
                a new new, thus if the name is unavailable an error is generated.
                However, this excluds variable names that are known.
            Specifying `alt_name` implies `find_new_name=False`.
            If you need to create a special array, you can use `jace.util.JaCeVar`
                to create a pseudo Jax variable.
        """
        assert self.is_allocated()

        shape: Sequence[int] = jutil.get_jax_var_shape(arg)
        dtype = jutil.get_jax_var_dtype(arg)
        offset = None  # i.e. no offset
        storage: dace.StorageType = dace.StorageType.Default  # Set at later stages (optimization)
        is_scalar: bool = shape == ()

        if (alt_name is None) and (self.map_jax_var_to_sdfg(arg, allow_fail=True) is not None):
            # Maybe the test could be more robust, but it will check if we try to create
            #  a variable for a second time. It is, however, okay to use one as template,
            #  if another name is specified from the beginning.
            raise ValueError(
                f"Tried to create variable '{arg}' again, without specifying an alternative name.."
            )
        if force_jax_name:
            if alt_name is not None:
                raise ValueError(
                    f"Specified 'force_jax_name', but passed '{alt_name}' as 'alt_name'."
                )
            if name_prefix is not None:
                raise ValueError(
                    f"Specified 'force_jax_name', but passed '{name_prefix}' as 'name_prefix'."
                )
            alt_name = jutil._propose_jax_name(arg, self._ctx.jax_name_map)
        if alt_name is not None:
            assert isinstance(
                alt_name, str
            ), f"Got '{type(alt_name)}' instead of 'str' for 'alt_name'."
            find_new_name = False  # If a name was given, then use it no matter what.
            if len(alt_name) == 0:
                raise ValueError("Passed an empty 'alt_name'.")
            if alt_name in self._forbidden_names:
                raise ValueError("'alt_name' is a forbidden name.")
            if not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", alt_name):
                raise ValueError(f"The passed name 'alt_name' '{alt_name}' is invalid.")
            if name_prefix is not None:
                raise ValueError(
                    f"Specified 'name_prefix' ('{name_prefix}') but passed '{alt_name}' as 'alt_name'."
                )
            if alt_name in self._ctx.sdfg.arrays:
                raise ValueError(f"Variable '{alt_name}' already exists.")
        if name_prefix is not None:
            assert isinstance(name_prefix, str)
            if len(name_prefix) == 0:
                raise ValueError("Specified an empty 'name_prefix'.")
        if as_view and (not as_transient):
            raise ValueError("You tried to create a global view, which is not allowed.")

        # Checking the strides.
        if (symb_strides is None) and (strides is None):
            def_symb_stride = False  # default value for symbolic strides
            symb_strides = False if (len(shape) <= 1) else def_symb_stride  # Keep for the future
        elif (symb_strides is not None) and (strides is not None):
            raise ValueError("Specified 'symb_strides' and 'stride at the same time.")
        elif strides is not None:
            if len(strides) != len(shape):
                raise ValueError(
                    f"'strides' has length {len(strides)}, but array rank is {len(shape)}."
                )
        else:
            assert isinstance(symb_strides, bool)

        # Now we determine the proposed name of the variable.
        #  Depending on the situation, we will further manipulate it.
        if alt_name is not None:
            prop_name = alt_name  # Just for completion: will be ignored later
        elif isinstance(arg, (jcore.Var, jutil.JaCeVar)):
            prop_name = jutil._propose_jax_name(arg, self._ctx.jax_name_map)
            if prop_name.startswith("__"):
                raise ValueError(
                    f"You tried to create the variable '{prop_name}' which"
                    "starts with two underscores, use 'alt_name' for that."
                )
            if name_prefix is not None:
                prop_name = name_prefix + prop_name
        elif isinstance(arg, jcore.Literal):  # type: ignore[unreachable]
            if not allow_literals:
                raise NotImplementedError("Jax Literals are not supported.")
            if alt_name is None:
                raise ValueError(f"Passed literal '{arg}', but not specified a name to use.")
        else:
            raise TypeError(f"Does not know how to handle '{type(arg).__name__}'.")
        if alt_name is None:
            # If we are the root translator, then we will use `prop_name` directly;
            #  otherwise we will append the revision of `self` to the name.
            arg_name = prop_name + (
                "" if self.is_root_translator() else f"_rev_idx{self._ctx.rev_idx}"
            )
        else:
            arg_name = str(alt_name)

        # Determine if we should look for a new name or not, if nothing was specified
        if find_new_name is None:
            if arg_name in self._reserved_names:
                find_new_name = True
            if arg_name in self._forbidden_names:
                find_new_name = True

        if find_new_name:
            # We have to find a new name.
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
        if not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", arg_name):
            raise ValueError(f"The requested variable name '{arg_name}' is invalid.")

        # Promotion of scalar to array.
        if is_scalar and force_array:
            shape = (1,)
            symb_strides = False
            strides = None
            is_scalar = False

        # Set the stride if we have to change.
        if strides is not None:
            strides = tuple(strides)
            assert len(strides) == len(shape)

        elif (symb_strides is True) and (not is_scalar):
            strides = [
                dace.symbol(f"{arg_name}_stride{dim}", dace.int64) if size >= 2 else 0
                for dim, size in enumerate(shape)
            ]

        if is_scalar:
            self._ctx.sdfg.add_scalar(
                name=arg_name, storage=storage, dtype=dtype, transient=as_transient
            )
        elif as_view:
            self._ctx.sdfg.add_view(
                name=arg_name,
                shape=shape,
                strides=strides,
                offset=offset,
                storage=storage,
                dtype=dtype,
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

    def create_jax_var_list(
        self,
        jax_var_list: Sequence[jcore.Atom | jutil.JaCeVar],
        prevent_creation: bool = False,
        only_creation: bool = False,
        handle_literals: bool = False,
        **kwargs: Any,
    ) -> list[None | str]:
        """Creates SDFG variables for the listed Jax variables and returns their SDFG names.

        If a Jax variable already has a SDFG equivalent then the function will use this variable.
        If no SDFG variable is known the function will create one using `add_array()`, with `update_var_mapping` set to `True`.

        By setting `prevent_creation` the function will not create any new SDFG variables.
        This mode is used to indicate that all variables already have to exists already.
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
        """
        if only_creation and prevent_creation:
            raise ValueError("Specified both 'only_creation' and 'prevent_creation'.")
        assert "update_var_mapping" not in kwargs

        ret_list: list[None | str] = []
        for jax_var in jax_var_list:
            if isinstance(jax_var, jcore.Literal):
                if not handle_literals:
                    raise ValueError("Encountered a literal but `handle_literals` was `False`.")
                sdfg_name = None
            elif isinstance(jax_var, (jcore.Var, jutil.JaCeVar)):
                mapped_sdfg_name: str | None = self.map_jax_var_to_sdfg(jax_var, allow_fail=True)
                if (mapped_sdfg_name is None) and prevent_creation:
                    raise ValueError(f"'prevent_creation' given but have to create '{jax_var}'.")
                if mapped_sdfg_name is None:
                    sdfg_name = self.add_array(arg=jax_var, update_var_mapping=True, **kwargs)
                elif only_creation:
                    raise ValueError(f"'only_creation' given '{jax_var}' already exists.")
                else:
                    sdfg_name = mapped_sdfg_name
                # `add_jax_name_mapping` is save, because if the mapping does already exists it is a no ops.
                self.add_jax_name_mapping(jax_var, sdfg_name)
            else:
                raise TypeError(f"Does not know how to handle '{type(jax_var).__name__}'")

            ret_list.append(sdfg_name)

        return ret_list

    def _create_initial_input(
        self,
        jaxpr: jcore.ClosedJaxpr,
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
        assert len(self._ctx.out_names) == 0

        # Handle the initial input arguments
        sdfg: dace.SDFG = self._ctx.sdfg
        init_in_var_names: Sequence[str] = self.create_jax_var_list(  # type: ignore[assignment]
            jax_var_list=jaxpr.jaxpr.invars,
            only_creation=True,
            as_transient=True,  # Explicit transient; no error!
            handle_literals=False,  # Initial arguments are never literals
            force_array=inp_scalar_as_array,
            force_jax_name=self.is_root_translator(),  # Ensure root get pure Jax names.
        )
        sdfg.arg_names.extend(init_in_var_names)

        # Store the list of inputs in self; this is done to simplify exporting.
        #  The output list is populated by `self._translate_jaxpr_internal()`
        self._ctx.inp_names = tuple(init_in_var_names)

        return init_in_var_names

    def _create_constants(
        self,
        jaxpr: jcore.ClosedJaxpr,
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
        if not len(jaxpr.consts):
            return []

        const_names: list[str] = []
        for cJaxVar, cValue in zip(jaxpr.jaxpr.constvars, jaxpr.consts, strict=False):
            c_sdfg_name = self.add_array(
                arg=cJaxVar,
                name_prefix="__const_",
                as_transient=True,
                symb_strides=False,
                strides=None,
                update_var_mapping=True,
            )
            # We have to pass the data descriptor to `add_constant()`, otherwise a new one would be created.
            self._ctx.sdfg.add_constant(
                c_sdfg_name, deepcopy(cValue), self._ctx.sdfg.arrays[c_sdfg_name]
            )
            const_names.append(c_sdfg_name)
        return const_names

    def _allocate_translation_ctx(
        self,
        name: str | None = None,
        reserved_names: str | Collection[str] | None = None,
    ) -> JaxprTranslationDriver:
        """This function allocates and initialize the members of the translation context of `self`.

        If this function is called and `self` is already allocated, the function will create a new context.
        This allows the driver to handle nested Jaxpr.
        The first context that is created is also known as root translator.

        Args:
            name:               The name of the SDFG.
            reserved_names:     Add these name to the set of resered names of `self`.
        """
        from ._translation_context import _TranslationContext

        if name and (not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", name)):
            raise ValueError(f"The provided name '{name}' for the SDFG is invalid.")

        # Create a new translation context and put it on the stack.
        self._ctx = _TranslationContext(
            rev_idx=next(self._rev_manager),
            name=name,
        )
        self._ctx_stack.append(self._ctx)

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

    def _init_sub_translators(
        self,
        subtrans_args: Mapping[str, Any],
    ) -> JaxprTranslationDriver:
        """This function initializes the subtranslator.

        The function forwards `kwargs` to the constructor of the subtranslators.
        However, it will remove all arguments starting with an underscore.
        """
        from jace.translator.sub_translators import _get_subtranslators_cls  # Avoid import cycle

        assert self._sub_translators is None

        subtrans_args = {k: v for k, v in subtrans_args.items() if not k.startswith("_")}  # type: ignore[unreachable]
        sub_translators: dict[str, jtrans.PrimitiveTranslator] = {}
        for sub_translator_cls in _get_subtranslators_cls():
            sub_translator: jtrans.PrimitiveTranslator = sub_translator_cls.CREATE(**subtrans_args)
            handled_primitives: Iterable[str] = jutil.as_sequence(
                sub_translator.get_handled_primitive()
            )
            for handled_primitive in handled_primitives:
                if handled_primitive in sub_translators:
                    raise RuntimeError(f"Multiple sub_translators for '{handled_primitive}' found.")
                sub_translators[handled_primitive] = sub_translator
        self._sub_translators = sub_translators

        return self

    def _clear_translation_ctx(self) -> JaxprTranslationDriver:
        """This function deallocate the translation context of `self`.

        Notes:
            While it is allowed for outside code to call this explicitly function,
                it is is most likely an error.
            If `self` is not allocated this function acts as a noops.
            The reserved names are only deallocated if `self` is a root translator.
        """
        if not self.is_allocated():
            return self

        assert self._ctx is self._ctx_stack[-1], "Inconsistent stack detected."
        if self.is_root_translator():
            self._rev_manager = itertools.count(0, 1)
            self._reserved_names = None  # type: ignore[assignment]

            self._ctx = None  # type: ignore[assignment]
            self._ctx_stack.pop()

        else:
            # Restore the previous state
            assert len(self._ctx_stack) > 1
            self._ctx_stack.pop()
            self._ctx = self._ctx_stack[-1]
        return self

    def _find_sub_translator_for(
        self,
        eqn: jcore.JaxprEqn,
    ) -> jtrans.PrimitiveTranslator:
        """Returns the appropriate subtranslator for equation `eqn`."""
        assert self._sub_translators is not None

        prim_name: str = eqn.primitive.name
        if prim_name not in self._sub_translators:
            raise NotImplementedError(f"No subtranslators known to handle '{prim_name}'.")

        return self._sub_translators[prim_name]

    def _translate_single_eqn(
        self,
        jaxpr: jcore.ClosedJaxpr,
        eqn: jcore.JaxprEqn,
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
            While `jaxpr` must be a `ClosedJaxpr`, `eqn` must come from the unclosed instance.
            The function will perform some consistency checking after the subtranslator was called.
        """
        assert isinstance(eqn, jcore.JaxprEqn)
        assert isinstance(jaxpr, jcore.ClosedJaxpr)

        if len(eqn.effects) != 0:
            raise NotImplementedError(f"Equation '{eqn}' has side effects.")

        # Input/Output variables
        #  Using a tuple for the input ensures that it is not modified.
        in_var_names: Sequence[str | None] = tuple(
            self.create_jax_var_list(
                eqn.invars,
                prevent_creation=True,  # Inputs must already exists.
                handle_literals=True,  #  but they can be literals.
            )
        )
        out_var_names: Sequence[str] = self.create_jax_var_list(  # type: ignore[assignment]
            eqn.outvars,
            only_creation=True,  # Output must not exist yet.
        )

        # Find the subtranslator
        subtranslator: jtrans.PrimitiveTranslator = self._find_sub_translator_for(eqn)

        # Create the state into which the equation should be translated
        last_term_state: dace.SDFGState = self.get_terminal_sdfg_state()  # noqa: F841 # Will be used later
        eqn_state = self.append_new_state(
            label=f"{eqn.primitive.name}_{out_var_names[0]}",
            prev_state=None,  # forces terminal state
        )

        # Now perform the actual translation of the equation.
        new_sdfg_term_state = subtranslator.translate_jaxeqn(
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
        elif isinstance(new_sdfg_term_state, dace.SDFGState):
            # TODO(phimuell): use `last_term_state` to test if `new_sdfg_term_state` is reachable.
            pass
        else:
            raise TypeError(f"Encountered illegal types '{type(new_sdfg_term_state)}'")

        # In case a subtranslator decided to not use the variables we created for it, which is allowed
        #  but he must update the `out_var_names` list correctly, we will now verify this.
        if len(out_var_names) != len(eqn.outvars):
            raise RuntimeError(
                f"Modified 'out_var_names'! Expected {len(eqn.outvars)} variables."
                f" but found {len(out_var_names)}"
            )
        for expectedSDFGName, jax_var in zip(out_var_names, eqn.outvars, strict=True):
            mapped_sdfg_name = self.map_jax_var_to_sdfg(jax_var)
            jax_name = jutil.get_jax_var_name(jax_var)
            if mapped_sdfg_name != expectedSDFGName:
                raise ValueError(
                    f"Mapping inconsistency detected, expected that Jax variable"
                    f" '{jax_name}' maps to '{expectedSDFGName}' but it actually"
                    f" maps to '{mapped_sdfg_name}'."
                )

        # Views can only be used if there is a direct connection, between source,
        #  view and destination (place of usage). Because of the way how Jax works,
        #  it is impossible that an output variable is a View.
        for outVarName, jax_var in zip(out_var_names, eqn.outvars, strict=True):
            sdfg_var = self.get_array(outVarName)
            if isinstance(sdfg_var, (dace.data.Array, dace.data.Scalar)):
                pass
            elif isinstance(sdfg_var, dace.data.View):
                raise TypeError(
                    f"For Jax variable '{jutil.get_jax_var_name(jax_var)}' (SDFG: '{outVarName}'),"
                    f" which is an output, you used a View, which is not possible."
                    " It must either be an array or a scalar."
                )
            else:
                raise NotImplementedError(
                    f"Output variable '{jutil.get_jax_var_name(jax_var)}' (SDFG: '{outVarName}')"
                    f" is of type '{type(sdfg_var).__name__}' which I does not know how to handle."
                )

        # Modify terminal root state of 'self'
        self._ctx.terminal_state = new_sdfg_term_state

        return (in_var_names, out_var_names)

    def _translate_jaxpr_internal(
        self,
        jaxpr: jcore.ClosedJaxpr,
    ) -> jtrans.TranslatedJaxprSDFG:
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
        assert isinstance(jaxpr, jcore.ClosedJaxpr)
        assert self.is_allocated()

        nb_translated_eqn: int = 0
        out_var_names: Sequence[str] = []
        for eqn in jaxpr.jaxpr.eqns:  # Translate the equations one by one.
            assert len(eqn.effects) == 0
            if len(eqn.outvars) == 0:  # Do we need this special case.
                continue  #  Looks more like internal Jax error.
            if any(jutil.is_drop_var(outVar) for outVar in eqn.outvars):
                assert (len(eqn.outvars) == 1) or all(
                    jutil.is_drop_var(outVar) for outVar in eqn.outvars
                )
                continue
            _, out_var_names = self._translate_single_eqn(jaxpr=jaxpr, eqn=eqn)
            nb_translated_eqn += 1

        if nb_translated_eqn == 0:
            # There were no equation, so handle the copying of input to output.
            out_var_names = self._handle_null_jaxpr(jaxpr)
        self._ctx.out_names = tuple(out_var_names)

        return self._export_context()

    def _export_context(self) -> jtrans.TranslatedJaxprSDFG:
        """Encapsulate the translation context of `self` into a `TranslatedJaxprSDFG` object..

        This function will not deallocate the internal context of `self`.
        Thus `self` and the return value will share the same context in memory.
        To free the context of `self` use `self._clear_translation_ctx()`.
        """
        assert self.is_allocated()
        assert all((isinstance(x, str) and (len(x) > 0)) for x in self._ctx.inp_names)
        assert all((isinstance(x, str) and (len(x) > 0)) for x in self._ctx.out_names)

        return jtrans.TranslatedJaxprSDFG(
            sdfg=self._ctx.sdfg,
            start_state=self._ctx.start_state,
            terminal_state=self._ctx.terminal_state,
            jax_name_map=self._ctx.jax_name_map,
            inp_names=self._ctx.inp_names,
            out_names=self._ctx.out_names,
        )

    def _handle_null_jaxpr(
        self,
        jaxpr: jcore.ClosedJaxpr,
    ) -> Sequence[str]:
        """This function is called in case a `Jaxpr` with zero equations is encountered.

        A function with zero equation might still have output, in which case an
        input is copied to an output. This function will handle the copying from
        the input into the corresponding output variable.

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

        # We will use this list to build the list of output names.
        #  This is important for the exporter.
        out_var_names: list[str] = []

        # If we are here then we are dealing with a nested SDFG/Jaxpr.
        #  Because an input also serves as output, the nested SDFG will have connector pairs
        #  with the same name, one serving as input the other as output, with the same name.
        #  This will make node validation fail.
        #  Thus we have to introduce a some fake output name and explicitly copy the data around.
        #  Once DaCe will inline the nested SDFG it will remove this intermediate copy.
        for jax_out_var in jaxpr.jaxpr.outvars:
            jax_inp_name = jutil.get_jax_var_name(
                jax_out_var
            )  # Since output == input their names must be the same.
            assert self.map_jax_var_to_sdfg(jax_inp_name, allow_fail=True)

            # This is the name we give to fictive Jax variable serving as output.
            jax_out_name = f"_zero_equation_output_{self.map_jax_var_to_sdfg(jax_out_var)}"

            # Now create the SDFG variable for it, give it a unique name.
            sdfg_out_name = self.add_array(
                jax_out_var,
                as_transient=True,
                name_prefix="_zero_equation_output_for_",
                update_var_mapping=False,
            )

            # We now create a new mapping, we do this that we will later find the variable again.
            self.add_jax_name_mapping(jax_var=jax_out_name, sdfg_name=sdfg_out_name)
            out_var_names.append(jax_out_name)

            # Now copy the input into the fake output variable.
            inp_acc = self._ctx.start_state.add_read(self.map_jax_var_to_sdfg(jax_inp_name))
            out_acc = self._ctx.start_state.add_write(self.map_jax_var_to_sdfg(jax_out_var))
            self._ctx.start_state.add_nedge(
                src=inp_acc,
                dst=out_acc,
                data=dace.Memlet.from_array(
                    jax_inp_name, self.get_array(self.map_jax_var_to_sdfg(jax_inp_name))
                ),
            )
        return tuple(out_var_names)

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
