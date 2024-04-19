# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Final, cast, overload

import dace
import jax
from dace import data as ddata, properties as dprop
from jax import core as jcore

from jace import translator
from jace.translator import util as jtrutil
from jace.util import jax as jutil


class JaxprTranslationDriver:
    """Internal driver class for creating an SDFG equivalent of a `Jaxpr` instance.

    The idea of the transformation is quite simple.
    Since Jaxpr is essentially a list consisting of more or less simple instructions, we will process them one after the other.
    For simplicity we will put each equation in its own state, primitives that needs more states must be put into a nested SDFG.

    This class builds an SDFG of a very particular form, which is not directly usable.
    But it is used as the canonical form inside JaCe and characterized by:
    - the SDFG is a list of states, each state corresponds to single Jax primitive,
    - all variable names are derived from Jax names,
    - there are no global variables inside the SDFG,
    - there is no possibility to return something.

    To support nested Jaxpr expressions the driver provides the possibility to clone/fork itself, see `self.fork()` for more.
    Clones, i.e. the return values of `self.fork()`, also known as children or clone, have a unique identifier, called revision.
    It is important that the revision is only unique within a family and during a translation process.
    This identifier is used to generate unique variable names.
    The clones form a tree that is rooted at the so called 'head translator', i.e. the driver that was explicitly created.

    The actual translation of a Jaxpr equation is not handled by the driver instance directly.
    Instead it is forwarded to a subtranslator instance, see `JaCeSubTranslatorInterface` for more.
    These subtranslators are independent objects that are owned by the driver.
    However, they are tightly coupled and thus a subtranslator is allowed to use the following private functions:
    - `_add_array()` if the translator has to create new.
    - `_create_jax_var_list()` for the bulk creation of Jax variables.
    - `_add_reserved_names()` if a name should be blocked (only affects later equation.
    - `_add_jax_name_mapping()` for creating new links between Jax variables and SDFG variables.
    However, a subtranslator should only call them if it is necessary.


    If no translation is ongoing the only function that makes sense to call is `translate_jaxpr()` to start a translation.
    Driver supplied to the subtranslators as arguments, such as in `translateEqn()` are allowed to call any public function of the driver.
    In addition to them it is allowed to call:

    Notes:
        Equations that only have `_` as output variable are skipped.
        It is not safe to deepcopy `self` during an active translation instead you should use `self.fork()`.
        To ensure unique names also in the presence of nested SDFG every instance contains a revision index.

    Todos:
        Split the functions into several interfaces one, that is for the whole world to use, one for subtranlators and one for the implementation.
    """

    # Member variables that are private to an instance, i.e. they are not passed on to the children.
    #  By definition all private variable belongs to the translation context but not all variable of the translation context are private.
    #   NOTE: The context also includes some shared members, but they are handled a bit differently.
    __private_slots__ = (
        "_sdfg",
        "_term_sdfg_state",
        "_init_sdfg_state",
        "_jax_name_map",
        "_sdfg_in_names",
        "_sdfg_out_names",
        "_rev_idx",
    )
    # These are the member variables that are shared among the forks.
    __shared_slots__ = (
        "_reserved_names",  # Part of the context.
        "_sub_translators",
        "_rev_manager",  # This is the revision counter manager
    )
    __slot__ = __private_slots__ + __shared_slots__

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Creates the base translator.

        This function will forward all arguments that does _not_ start with an underscore to the constructors of the subtranslators.
        Furthermore, this function will allocate the shared members, but the private members are not allocated.

        Args:
            _no_shared_alloc (bool):     If set then all allocation will be avoided (internal)

        Notes:
            All arguments that does not start with an underscore are forwarded to the translators for the intrinsics.
            By setting `_no_shared_alloc` to `True` the function will not allocate the shared part.
                This flag is provided only for implementing `self.fork()` using it denotes an error and undefined behaviour.
        """
        allocate_shared_parts: bool = not kwargs.pop("_no_shared_alloc", False)

        # Contains all the subtranslators that we need.
        #  They are partitioned by the names of the primitive they have registered for.
        #  Inside a partition they are ordered by priority, lowest first, more important.
        #  This member is allocated by '_init_sub_translators()' and remains allocated during the lifetime of the object.
        self._sub_translators: dict[str, list[translator.JaCeSubTranslatorInterface]] = None  # type: ignore[assignment]

        # The SDFG object that we are currently constructing.
        #  Only allocated during an ongoing translation.
        self._sdfg: dace.SDFG = None

        # This is the HEAD SDFG state, i.e. the last state in which we translated an equation.
        #  Only allocated during an ongoing translation.
        self._term_sdfg_state: dace.SDFGState = None

        # This is the beginning of the SDFG, i.e. the original SDFG HEAD.
        #  Only allocated during an ongoing translation.
        self._init_sdfg_state: dace.SDFGState = None

        # This is the mapping, that maps the Jax name to the name that is used inside the SDFG.
        #  Only allocated during an ongoing translation.
        self._jax_name_map: dict[str, str] = None  # type: ignore[assignment]

        # These names can not be used for the automatic naming of Jax variables.
        #  They differ from the forbidden names, that they denote valid SDFG names.
        #  An example would be names of the function arguments.
        #  Only allocated during an ongoing translation.
        self._reserved_names: set[str] = None  # type: ignore[assignment]

        # These are the names of the SDFG variables that serves as input and output.
        #  They have the same order as in the Jaxpr.
        #  Only allocated during an ongoing translation.
        self._sdfg_in_names: Sequence[str] = None  # type: ignore[assignment]
        self._sdfg_out_names: Sequence[str] = None  # type: ignore[assignment]

        # This is the manager for the revision counter.
        #  It is shared among all children.
        #  Might be overwritten if we are in the context of 'fork()'.
        self._rev_manager: jtrutil.RevisionCounterManager = jtrutil.RevisionCounterManager()

        # This is the revision of self.
        #  Unlike the manager it is not shared and private.
        #  Might be overwritten in the context of a fork.
        self._rev_idx: int = self._rev_manager.assign_revision()
        assert self.is_head_translator()

        # If requested we will now allocate some internal state
        if allocate_shared_parts:
            self._init_sub_translators(kwargs)

    def translate_jaxpr(
        self,
        jaxpr: jcore.ClosedJaxpr,
        *,
        inp_scalar_as_array: bool = False,
        name: str | None = None,
        reserved_names: str | Collection[str] | None = None,
        allow_empty_jaxpr: bool = False,
        _clear_translation_ctx: bool = True,
    ) -> jtrutil.JaCeTranslationMemento:
        """Perform the translation of a Jaxpr description into a SDFG.

        As described above the function will create the canonical form of Jaxpr based SDFGs.
        Furthermore the function will return the SDFG encaplulated inside a `jace.translator.util.JaCeTranslationMemento` object.

        Args:
            inp_scalar_as_array:       Translate scalar _input_ arguments to arrays of length 1.
            name:                   Use this name for the SDFG instead some generated one.
            reserved_names:          Prevent the generation of such names, when translating Jax variable names into SDFG names.
            allow_empty_jaxpr:        Allows empty Jaxpr.
            _clear_translation_ctx:      Do not deallocate the inner state of `self`.

        Notes:
            By default the function will store its translation state inside the return value and deallocate the internal members.
                However, by setting `_clear_translation_ctx` to `False` `self` is not deallocated.
                This means that `self` and the returned memento share the same state.
                To explicitly deallocate the translation context of `self`, which is required, use `self._clearTranslatorCtx()`.
        """
        if self.is_allocated():
            raise RuntimeError(
                "The translator driver is already allocated, you should resort to 'fork()'."
            )
        if (len(jaxpr.eqns) == 0) and (not allow_empty_jaxpr):
            raise ValueError("Passed an empty Jaxpr, but did not allow for empty Jaxpr.")
        if not isinstance(jaxpr, jcore.ClosedJaxpr):
            raise TypeError(f"Expected a 'jax.core.ClosedJaxp' instance but got '{type(jaxpr)}'")
        if len(jaxpr.effects) != 0:
            raise NotImplementedError(
                "Currently 'Jaxpr' instances with side effects are not supported."
            )
        if len(jaxpr.out_avals) == 0:
            raise ValueError("Jaxpr has zero output variables.")
        if not jax.config.read("jax_enable_x64"):
            raise NotImplementedError(
                "The translation only works if 'jax_enable_x64' is enabled. Do it manually or use 'self.transform()'!"
            )

        self._allocate_translation_ctx(
            name=name,
            reserved_names=reserved_names,
        )
        self._create_initial_input(
            jaxpr=jaxpr,
            inp_scalar_as_array=inp_scalar_as_array,
        )
        self._create_constants(
            jaxpr=jaxpr,
        )
        memento: jtrutil.JaCeTranslationMemento = self._translate_jaxpr_internal(jaxpr)

        if _clear_translation_ctx:
            self._clear_translation_ctx()

        return memento

    def fork(self) -> JaxprTranslationDriver:
        """Return a child of `self` ready for transformation.

        The returned object, known as child, will always be of type `JaxprTranslationDriver`, and should be seen as a partial clone of `self`.
        While the child shares some members with its parent, i.e. `self`, it has an unallocated translation context.
        Essentially, this function returns an object that when its `translate_jaxpr()` function is called behaves the exact same way as
        its parent behaved as it was called just with another `jaxpr` argument.

        Notes:
            A user has to ensure that the lifetime of a fork ends before the one of its direct parent.
                In case of a head translator, the lifetime of its children have to end before the translation process finishes.
        """
        # Create a new (empty) driver instance; prevent allocation to make it cheep
        dolly: JaxprTranslationDriver = JaxprTranslationDriver(_no_shared_alloc=True)

        # Copy the shared members from parent to fork.
        for slot_name in self.__shared_slots__:
            setattr(dolly, slot_name, getattr(self, slot_name))

        # Handle the special members and initialize them.
        dolly._rev_idx = dolly._rev_manager.assign_revision()
        assert not dolly.is_head_translator()

        return dolly

    def append_new_state(
        self,
        label: str | None = None,
        condition: dprop.CodeBlock | None = None,
        assignments: Mapping[str, Any] | None = None,
        *,
        prev_state: dace.SDFGState | None = None,
    ) -> dace.SDFGState:
        """Creates a new SDFGState and appends it.

        By default the new SDFGState is appended to the current terminal SDFGState.
        However, if `prev_state` is given the new SDFGState will be appended to it instead.

        Args:
            label:          The name that should be used for the new SDFGState.
            condition:      The condition of the state transitions used on the InterstateEdge.
            assignments:    Symbol assignments that should be done during the transition.
            prev_state:      Alternative SDFGState to which we should append the new SDFGState.

        Notes:
            In case no SDFGState exists yet, an initial SDFGState will be created first.
            This function is similar to `SDFGState.add_state_after()` but differs in the fact that it does not perform reconnecting.
                I.e. if the state to which we append already has downstream states they will not be reconnected to be after the newly created state.
            This function will not update the head state of `self`.
        """
        assert self._sdfg is not None

        # Test if we must create a start state.
        if self._sdfg.start_block is None:
            self._init_sdfg_state = self._sdfg.add_state(label="initial_state", is_start_block=True)
            self._term_sdfg_state = self._init_sdfg_state
        assert self._sdfg.start_block is self._init_sdfg_state

        # Now create and append the new state
        app_state: dace.SDFGState = self._term_sdfg_state if prev_state is None else prev_state
        new_state = self._sdfg.add_state(label, is_start_block=False)
        self._sdfg.add_edge(
            app_state,
            new_state,
            dace.sdfg.InterstateEdge(condition=condition, assignments=assignments),
        )

        return new_state

    def get_arrays(self) -> Mapping[str, ddata.Data]:
        """Get the maps containing all known arrays inside the SDFG.

        Essentially a shorthand and preferred way for `self.get_sdfg().arrays`.
        """
        assert self._sdfg is not None
        return cast(Mapping[str, ddata.Data], self._sdfg.arrays)

    def get_array(
        self,
        name: str | jcore.Atom,
    ) -> ddata.Data:
        """Returns the `dace.data.Data` object `name` referees to.

        If `name` is a string, it is directly interpreted as the name of an SDFG variable.
        In case it is a `jax.core.Atom` it is first translated.
        """
        assert self._sdfg is not None

        if isinstance(name, str):
            pass
        elif isinstance(name, jcore.Atom):
            name = self.map_jax_var_to_sdfg(name)
        else:
            raise TypeError(f"Does not know how to handle '{type(name).__name__}'.")
        if name not in self._sdfg.arrays:
            raise KeyError(f"Requested the SDFG array '{name}' but it is not known.")

        return self._sdfg.arrays[name]

    @overload
    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jcore.Atom,
    ) -> str: ...

    @overload
    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jcore.Atom,
        allow_fail: bool,
    ) -> str | None: ...

    def map_jax_var_to_sdfg(
        self,
        jax_var: str | jcore.Atom,
        allow_fail: bool = False,
    ) -> str | None:
        """Returns the name of the SDFG variable that the Jax variable `jax_var` is referring to.

        Args:
            jax_var:        The Jax variable to look up.
            allow_fail:     If mapping is not known return `None` instead of raise `KeyError`.
        """
        assert self._jax_name_map is not None
        assert isinstance(jax_var, (jcore.Atom, str))

        jax_var = jutil.get_jax_var_name(jax_var)
        if jax_var not in self._jax_name_map:
            if allow_fail:
                return None
            KeyError(f"The Jax variable '{jax_var}' was never registered.")

        return self._jax_name_map[jax_var]

    def get_sdfg(self) -> dace.SDFG:
        """Returns the tentative SDFG that is currently constructed.

        If you want access to the arrays of the SDFG use `self.get_arrays()`/`self.get_array()`.
        """
        assert self._sdfg is not None
        assert (self._init_sdfg_state is None) or (self._init_sdfg_state is self._sdfg.start_block)
        return self._sdfg

    def get_terminal_sdfg_state(self) -> dace.SDFGState:
        """Returns the current tentative terminal state of the SDFG under construction.

        Since the translator works by turning each Jax primitive into an SDFG state, the constructed SDFG is essentially a list of states.
        This function returns the tentative final/terminal SDFGState of the SDFG.
        States of new primitives will be appended to this one.

        Notes:
            It is an error to call this function outside the context of a subtranslator.
                If you want access to the arrays of the SDFG use `self.get_arrays()`.
        """
        assert all(x is not None for x in (self._sdfg, self._term_sdfg_state))
        return self._term_sdfg_state

    def is_allocated(self) -> bool:
        """Tests if `self` is allocated.

        This function only checks if the translation context is allocated.
        As a side effect a return value of `True` means that a translation process is ongoing.

        Notes:
            The state of the reserved name list is handled specially.
                In case the function returns `True` it is guaranteed that it is allocated.
                If `False` is returned it might or might not be allocated.
        """
        small_ctx: Sequence[Any] = [
            getattr(self, x) for x in self.__shared_slots__ if x != "_reserved_names"
        ]
        if all((x is not None) for x in small_ctx):
            if self._reserved_names is None:
                raise RuntimeError(
                    "Invalid allocation state: All context variables except the reserved name list are allocated."
                )
            return True
        if all((x is None) for x in small_ctx):
            return False
        raise RuntimeError("Invalid allocation state: Translation context is mixed allocated.")

    def is_head_translator(self) -> bool:
        """Tests if `self` is a head translator.

        A head translator is a translator/driver that was created explicitly, i.e. not by `self.fork()`.
        """
        return self._rev_manager.is_root_revision(self._rev_idx)

    def same_family(
        self,
        other: JaxprTranslationDriver,
    ) -> bool:
        """Test if `self` and `other` belongs to the same family of driver/translators.

        They belong to the same family if they descend from the same head translator.
        """
        if not isinstance(other, JaxprTranslationDriver):
            return NotImplemented  # type: ignore[unreachable]
        if all(getattr(self, x) is getattr(self, x) for x in self.__shared_slots__):
            assert (self if (self._rev_idx < other._rev_idx) else other).is_allocated()
            return True
        assert not any(getattr(self, x) is getattr(self, x) for x in self.__shared_slots__)

        return False

    @staticmethod
    def translate_dtype(dtype: Any) -> dace.typeclass:
        """Turns a Jax datatype into a DaCe datatype.

        Todo:
            Improve.
        """
        nameof_dtype = str(dtype)

        # Make some basic checks if the datatype is okay
        if (not jax.config.read("jax_enable_x64")) and (nameof_dtype == "float64"):
            raise ValueError("Found a 'float64' type but 'x64' support is disabled.")
        if nameof_dtype.startswith("complex"):
            raise NotImplementedError("Support for complecx computation is not implemented.")

        # Now extract the datatype from dace, this is extremely ugly.
        if not hasattr(dace.dtypes, nameof_dtype):
            raise TypeError(
                f"Could not find the type '{nameof_dtype}' ({type(dtype).__name__}) in 'dace.dtypes'."
            )
        dcd_type = getattr(dace.dtypes, nameof_dtype)

        if not isinstance(dcd_type, dace.dtypes.typeclass):
            raise TypeError(
                f"Expected that '{nameof_dtype}' would map to a 'dace.typeclass' but it mapped to a '{type(dcd_type).__name__}'."
            )

        return dcd_type

    def _add_jax_name_mapping(
        self, jax_var: str | jcore.Atom, sdfg_name: str
    ) -> JaxprTranslationDriver:
        """Creates the mapping between `jax_var` to `sdfg_name`.

        It is an error if there is already a mapping installed for `jax_var`.

        Args:
            jax_var:     The Jax variable that is used.
            sdfg_name:   The name of the corresponding SDFG variable.

        Notes:
            While the function allows to create a mapping for Jax names that are in the set of avoided names,
                it will refuse to create a mapping for a forbidden name.
        """
        assert self._jax_name_map is not None
        assert isinstance(jax_var, (jcore.Atom, str))

        jax_name = jutil.get_jax_var_name(jax_var)
        if jax_name in self._jax_name_map:
            if self._jax_name_map[jax_name] == sdfg_name:  # We consider this as no ops.
                return self
            raise ValueError(
                f"Tried to create a mapping for Jax variable '{jax_name}' to '{sdfg_name}', but that mapping exists already and is pointing to '{self.map_jax_var_to_sdfg(jax_name)}'."
            )
        if sdfg_name not in self.get_arrays():
            raise KeyError(
                f"Tried to create the mapping '{jax_name} -> {sdfg_name}', but '{sdfg_name}' is not a known SDFG variable."
            )
        if sdfg_name in self._forbidden_names:
            raise NameError(  # This is actually an internal error
                f"Tried to create the mapping '{jax_name} -> {sdfg_name}', but '{sdfg_name}' is forbidden."
            )

        self._jax_name_map[jax_name] = sdfg_name
        return self

    def _add_reserved_names(
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
            raise TypeError(
                f"Does not know how to handle the type '{type(reserved_names).__name__}'."
            )
        assert all(isinstance(x, str) for x in reserved_names)

        self._reserved_names.update(reserved_names)
        return self

    def _add_array(
        self,
        arg: jcore.Atom,
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
        """Creates an SDFG variable for the Jax variable `arg` and returns the SDFG name.

        By default the function will create a transient, which can be changed by setting `as_transient` to `False`.
        In case the Jax variable `arg` refers to a scalar, i.e. having an empty shape, the function will generate a SDFG scalar.
        However, if `force_array` is set, then it will generate an array with shape `(1,)`.
        For generating a `View` you must set `as_view` to `True`.

        By specifying `alt_name` it is possible to force a certain name on a variable.
        It is important that if `alt_name` is specified the function will either generate the variable or fail.
        In case `alt_name` is not given, then the function will be derived one from `jutil.get_jax_var_name(arg)`.
        The  driver distinguishes between two kinds of "bad (SDFG) variable names".
        The first category are the forbidden names, which the function refuses to generate.
        The second one are the reserved names, which were set at the beginning.
        These names can be used if they are specified through `alt_name` but are not used in automatic naming.

        If nothing is specified, the strides of the data are determined by DaCe, which is continuous C order.
        There are two ways to change that.
        The first way is to specify the `strides` argument, which are then forwarded to the underlying DaCe function.
        The function will only check if enough values were provided, but no further check is performed.
        The second one is to set `symb_strides` to `True` in which case the function will generate symbols and use them.
        However, even if symbolic strides are activated, arrays with just one dimensions have always a non symbolic stride.
        Furthermore, dimensions with shape 1 will always have stride 0.

        By default this function does not update the internal variable map.
        However, by setting `update_var_mapping` to `True` the function will update the mapping.

        Args:
            arg:                The Jax object for which a SDFG equivalent should be created.
            as_transient:        If set, the SDFG variable is a transient, `True` by default.
            alt_name:            Try to create the variable with this name; either succeed or fail.
            name_prefix:         If given and in automatic naming mode, add this prefix to the name before anything else.
            force_array:         Instead of a `dace.Scalar` object create a `dace.Array` object with one element.
            as_view:             Creates a view instead of an array, if it is a scalar it is silently ignored.
            strides:            Instead of the default strides use this value for the strides.
            symb_strides:        Create symbols and use them for fully symbolic strides.
            find_new_name:        The translator will try to find a new name if the designated is already occupied.
                                    This does not work if the name was supplied by `alt_name`.
            allow_literals:      If `True` then also allows JaxLiterals as `arg`.
            force_jax_name:       If `True` then, the verbatim Jax name will be used.
            update_var_mapping:   Update the internal variable mapping; by default `False`.

        Notes:
            If `find_new_name` is `None` the default the function will only look for a new name if there is a need for that.
                If it is `True` the function will always look for a new name, even if the initial name was fine.
                If it is `False` the function will never look for a new new, thus if the name is unavailable an error is generated.
            Specifying `alt_name` implies `find_new_name=False`.
            The effect of specifying `force_jax_name` is as passing `jutil.get_jax_var_name(arg)` as `alt_name`.
        """
        assert all(x is not None for x in (self._sdfg, self._jax_name_map))
        shape: Sequence[int] = arg.aval.shape  # Shape of the array
        offset = None  # i.e. no offset
        storage: dace.StorageType = dace.StorageType.Default  # Set at later stages (optimization)
        is_scalar: bool = shape == ()
        dtype = self.translate_dtype(arg.aval.dtype)

        if (alt_name is not None) and (not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", alt_name)):
            raise ValueError(f"The passed name 'alt_name' '{alt_name}' is invalid.")

        if force_jax_name:
            if alt_name is not None:
                raise ValueError(
                    f"Specified 'force_jax_name' but passed '{alt_name}' as 'alt_name'."
                )
            if name_prefix is not None:
                raise ValueError(
                    f"Specified 'force_jax_name' and set 'name_prefix' to '{name_prefix}'."
                )
            alt_name = jutil.get_jax_var_name(arg)
        if name_prefix is not None:
            assert isinstance(name_prefix, str)
            assert len(name_prefix) > 0
            if alt_name is not None:
                raise ValueError("Specified 'name_prefix' and 'alt_name' which is not possible.")

        if (symb_strides is None) and (strides is None):
            symb_strides = False if (len(shape) <= 1) else False
        if as_view and (not as_transient):
            raise ValueError("You tried to create a global view, which is not allowed.")

        if isinstance(arg, jcore.Var):
            prop_name = jutil.get_jax_var_name(
                arg
            )  # This is the name that is _suggested_ by the conversion.
            if (alt_name is None) and prop_name.startswith("__"):
                raise ValueError(
                    f"You tried to create the variable '{prop_name}' which starts with two underscores, if you really want to do that use 'alt_name'."
                )
            if isinstance(name_prefix, str):
                prop_name = name_prefix + prop_name
        elif isinstance(arg, jcore.Literal):
            if not allow_literals:
                raise NotImplementedError("Jax Literals are not yet implemented.")
            if alt_name is None:
                raise ValueError(f"Passed literal '{arg}', but not specified a name to use.")
        else:
            raise TypeError(f"Does not know how to handle '{type(arg).__name__}'.")

        if alt_name is None:
            # If we are the root translator, then we will use `prop_name` directly;
            #  if not we will append the revision of `self` to the name.
            arg_name = prop_name + ("" if self.is_head_translator() else f"_rev_idx{self._rev_idx}")
        else:
            arg_name = str(alt_name)
            find_new_name = False  # If a name was given, then use it no matter what.
            if arg_name in self._forbidden_names:
                raise ValueError(f"You used 'alt_name' to create the forbidden name '{alt_name}'.")
            if arg_name in self._sdfg.arrays:
                raise ValueError(
                    f"Tried to create a variable with name '{arg_name}' explicitly, but it is already known."
                )
        if find_new_name is None:
            find_new_name = (arg_name in self._forbidden_names) or (
                arg_name in self._reserved_names
            )

        if find_new_name:
            # We have to find a new name.
            name_tmpl = "_jax_variable__" + arg_name + "__{}"
            for iCounter in range(1000):
                _arg_name = name_tmpl.format(iCounter)
                if (
                    (_arg_name in self._forbidden_names)
                    or (_arg_name in self._reserved_names)
                    or (_arg_name in self._sdfg.arrays)
                ):
                    continue  # The proposed variable is known, so try next value.
                arg_name = _arg_name  # We found a name that we can use.
                break
            else:
                raise ValueError(f"Failed to find a replacement name for '{arg_name}'")
            del iCounter, _arg_name
        elif arg_name in self._forbidden_names:
            raise ValueError(f"Can not create variable '{arg_name}', name is forbidden.")
        elif arg_name in self._sdfg.arrays:
            raise ValueError(f"Can not create variable '{arg_name}', variable is already created.")
        if not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", arg_name):
            raise ValueError(f"The requested variable name '{arg_name}' is invalid.")

        # Promotion of scalar to array.
        if is_scalar and force_array:
            shape = (1,)
            symb_strides = False
            strides = None
            is_scalar = False

        if strides is not None:
            if symb_strides:
                raise ValueError("Specified 'symb_strides' and 'stride at the same time.")
            if len(strides) != len(shape):
                raise ValueError(
                    f"'strides' was '{strides}' it had length {len(strides)}, but the array has rank {len(shape)}."
                )
            strides = tuple(strides)

        elif (symb_strides is True) and (not is_scalar):
            strides = [
                dace.symbol(f"{arg_name}_stride{dim}", dace.int64) if size >= 2 else 0
                for dim, size in enumerate(shape)
            ]

        if is_scalar:
            self._sdfg.add_scalar(
                name=arg_name, storage=storage, dtype=dtype, transient=as_transient
            )
        elif as_view:
            self._sdfg.add_view(
                name=arg_name,
                shape=shape,
                strides=strides,
                offset=offset,
                storage=storage,
                dtype=dtype,
            )
        else:
            self._sdfg.add_array(
                name=arg_name,
                shape=shape,
                strides=strides,
                offset=offset,
                storage=storage,
                dtype=dtype,
                transient=as_transient,
            )

        if update_var_mapping:
            self._add_jax_name_mapping(jax_var=arg, sdfg_name=arg_name)

        return arg_name

    def _create_jax_var_list(
        self,
        jax_var_list: Sequence[jcore.Atom],
        prevent_creation: bool = False,
        only_creation: bool = False,
        **kwargs: Any,
    ) -> list[None | str]:
        """Creates SDFG variables for the listed Jax variables and returns the SDFG names as a list.

        Before the function will create a variable, by using `_add_array()` with `update_var_mapping=True`,
        the function will check if the variable is known and no new variable is created.
        Instead the name of the previously created variable is added to the return value.
        In case the Jax Atom denotes a literal, no variable will be created, instead `None`
        will be added to the output list.

        Args:
            jax_var_list:         The list of Jax variables that should be transformed to SDFG names.
            prevent_creation:    Never create a variable, indicates that all variables must already exists.
            only_creation:       Indicates that no variables exists yet and all must be created.
            kwargs:             In case of variable creation will be forwarded to `self._add_array()` function.

        Notes:
            Expected input arguments are `jcore.JaxprEqn.invars` or `jcore.JaxprEqn.outvars`.
            If `only_creation` is set, then literals will cause an error.
            It is an error to pass the `update_var_mapping` argument.
        """
        assert self._jax_name_map is not None
        if only_creation and prevent_creation:
            raise ValueError("Specified both 'only_creation' and 'prevent_creation'.")

        ret_list: list[None | str] = []
        for jax_var in jax_var_list:
            if isinstance(jax_var, jcore.Literal):
                if only_creation:
                    raise ValueError(f"Requested 'only_creation', but '{jax_var}' is a 'Literal'.")
                ret_list.append(None)
            elif isinstance(jax_var, jcore.jax_var):
                mapped_sdfg_name: str | None = self.map_jax_var_to_sdfg(jax_var, allow_fail=True)
                if mapped_sdfg_name is None:
                    if prevent_creation:
                        raise ValueError(
                            f"Forbid the creation of jaxVariables, but need to create '{jax_var!s}'."
                        )
                    ret_list.append(self._add_array(arg=jax_var, update_var_mapping=True, **kwargs))
                else:
                    if only_creation:
                        raise ValueError(
                            f"Requested 'only_creation', but '{jax_var}' already exists as '{mapped_sdfg_name}'."
                        )
                    ret_list.append(mapped_sdfg_name)
            else:
                raise ValueError(
                    f"The translation process is not implemented for '{type(jax_var)}'"
                )

        return ret_list

    def _create_initial_input(
        self,
        jaxpr: jcore.ClosedJaxpr,
        inp_scalar_as_array: bool,
    ) -> Sequence[str]:
        """This function will create the internal input variables that are used for the SDFG.

        Args:
            jaxpr:              The Jaxpr that we want to translate.
            inp_scalar_as_array:   Promote scalars to arrays of size one.

        Returns:
            The list of SDFG variables used as input arguments of `jaxpr` in the same order.

        Notes:
            This function will fill the internal list of inputs.
        """
        assert self.is_allocated()
        assert len(jaxpr.jaxpr.invars)

        if len(self._sdfg_in_names) != 0:
            raise RuntimeError("Called '_create_initial_input()' twice?")
        assert len(self._sdfg_out_names) == 0

        # Handle the initial input arguments
        sdfg: dace.SDFG = self._sdfg
        init_in_var_names: Sequence[str] = self._create_jax_var_list(  # type: ignore[assignment]
            jax_var_list=jaxpr.jaxpr.invars,
            only_creation=True,
            as_transient=True,  # Explicit transient; no error!
            force_array=inp_scalar_as_array,
            force_jax_name=self.is_head_translator(),  # Ensure head get the pure Jax name.
        )
        sdfg.arg_names.extend(init_in_var_names)

        # Store the list of inputs in self; this is done to simplify exporting.
        #  The output list is either generated by `self._translate_jaxpr_internal()` of `self._handle_null_jaxpr()`.
        self._sdfg_in_names = tuple(init_in_var_names)

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

        assert self.is_allocated()
        if not len(jaxpr.consts):
            return []

        const_names: list[str] = []
        for cJaxVar, cValue in zip(jaxpr.jaxpr.constvars, jaxpr.consts, strict=False):
            c_sdfg_name = self._add_array(
                arg=cJaxVar,
                name_prefix="__const_",
                as_transient=True,
                symb_strides=False,
                strides=None,
                update_var_mapping=True,
            )
            # We have to pass the data descriptor to `add_constant()`, otherwise a new one would be created.
            self._sdfg.add_constant(c_sdfg_name, deepcopy(cValue), self._sdfg.arrays[c_sdfg_name])
            const_names.append(c_sdfg_name)
        return const_names

    def _allocate_translation_ctx(
        self,
        name: str | None = None,
        reserved_names: str | Collection[str] | None = None,
    ) -> JaxprTranslationDriver:
        """This function allocates and initialize the members related to the translation context.

        After this function is called, `self` is said to have an ongoing translation process.

        Args:
            name:               The name of the SDFG.
            reserved_names:     Add these name to the set of resered names of `self`.

        Notes:
            It is not an error, if the reserved names are already allocated.
                In that case the names passed by `reserved_names` are added to the list already preset.
        """
        if self.is_allocated():
            raise RuntimeError("The translator is already allocated.")
        if name and (not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", name)):
            raise ValueError(f"The provided name '{name}' for the SDFG is invalid.")

        self._sdfg = dace.SDFG(name=(name or f"unnamed_SDFG_{id(self)}"))
        self._init_sdfg_state = self._sdfg.add_state(label="initial_state", is_start_block=True)
        self._term_sdfg_state = self._init_sdfg_state
        self._jax_name_map = {}
        self._sdfg_in_names = ()
        self._sdfg_out_names = ()

        # Handle the `reserved_names` argument as described above.
        #  This is essentially needed that children works properly.
        if self._reserved_names is None:
            self._reserved_names = set()  # type: ignore[unreachable]
        else:
            raise RuntimeError("The reserved names are allocated incorrectly.")
        assert all(isinstance(x, str) for x in self._reserved_names)  # type: ignore[unreachable]
        self._add_reserved_names(reserved_names)

        return self

    def _init_sub_translators(
        self,
        kwargs: Mapping[str, Any],
    ) -> JaxprTranslationDriver:
        """This function initializes the subtranslator.

        The function forwards `kwargs` to the constructor of the subtranslators.
        However, it will remove all arguments starting with an underscore.
        """
        if isinstance(self._sub_translators, dict):
            raise RuntimeError("Tried to allocate the internal subtranslators twice.")
        assert self._sub_translators is None  # type: ignore[unreachable]

        # We might get arguments that starts with an underscore, which are not meant for the subtranslators.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

        # Will contain all subtranslators we create.
        subtranslators: dict[str, list[translator.JaCeSubTranslatorInterface]] = {}

        # First we will create all subtranslators and partition them.
        subtranslator_cls: type[translator.JaCeSubTranslatorInterface]
        for subtranslator_cls in []:
            subtranslator: translator.JaCeSubTranslatorInterface = subtranslator_cls(**kwargs)
            handled_primitives: Iterable[str] = jutil.ensure_iterability(
                subtranslator.getHandledPrimitives()
            )

            # Now add the subtranslator to the primitives it requests, we will sort them later into the correct order.
            for handledPrimitive in handled_primitives:
                subtranslators.setdefault(handledPrimitive, []).append(subtranslator)

        # Now we order the subtranslators for the primitives.
        self._sub_translators = {
            prim_name: jtrutil.sort_subtranslators(primSubTranslators)
            for prim_name, primSubTranslators in subtranslators.items()
        }
        return self

    def _clear_translation_ctx(self) -> JaxprTranslationDriver:
        """This function deallocate the translation context of `self`.

        Notes:
            While it is allowed for outside code to call this explicitly function, it is is most likely an error.
            If this function is called on a head translator, then the revision state will be rested.
                Thus a caller has to make sure that the lifetime of all children has ended.
            If `self` is not allocated this function acts as a noops.
            The reserved names are only deallocated if `self` is a head translator.
        """
        if not self.is_allocated():
            return self
        self._sdfg = None
        self._init_sdfg_state = None
        self._term_sdfg_state = None
        self._jax_name_map = None  # type: ignore[assignment]
        self._sdfg_in_names = None  # type: ignore[assignment]
        self._sdfg_out_names = None  # type: ignore[assignment]

        if self.is_head_translator():
            # We are the head translator thus we reset the revision manager.
            #  Since this function is only called at the very end, we know that the translation process as a whole has finished.
            #  We reset the state that the numbers are small again when we start anew.
            self._rev_manager._reset_state()

            # Freeing the reserved names only for heads make it more safe in case a child translator is reused.
            #  On the other hand reusing a child translator is discouraged, but not forbidden.
            self._reserved_names = None  # type: ignore[assignment]
        return self

    def _find_sub_translator_for(
        self,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jcore.JaxprEqn,
    ) -> translator.JaCeSubTranslatorInterface:
        """Returns the subtranslator object to translate `eqn`.

        The subtranslators are checked for applicability in the order of their priority.
        The fist one that accepts the translation will be taken.

        Notes:
            The arguments are the same as for `JaCeSubTranslatorInterface.can_translate_jaxeqn()`.
        """
        assert self._sub_translators is not None

        prim_name: str = eqn.primitive.name
        if prim_name not in self._sub_translators:
            raise NotImplementedError(f"No subtranslators known to hanble primitive '{prim_name}'.")
        subtranslator_canidates = self._sub_translators[prim_name]
        assert len(subtranslator_canidates) > 0

        subtranslator: translator.JaCeSubTranslatorInterface = None  # type: ignore[assignment]
        if len(subtranslator_canidates) == 1:
            subtranslator = next(iter(subtranslator_canidates))
            assert subtranslator.can_translate_jaxeqn(
                driver=self, in_var_names=in_var_names, out_var_names=out_var_names, eqn=eqn
            )
        else:
            for subtranslatorCanidate in subtranslator_canidates:
                if subtranslatorCanidate.can_translate_jaxeqn(
                    driver=self,
                    in_var_names=in_var_names,
                    out_var_names=out_var_names,
                    eqn=eqn,
                ):
                    subtranslator = subtranslatorCanidate
            else:
                raise NotImplementedError(f"No subtranslator found for handling '{eqn}'.")
        return subtranslator

    def _translate_single_eqn(
        self,
        jaxpr: jcore.ClosedJaxpr,
        eqn: jcore.JaxprEqn,
    ) -> tuple[Sequence[str | None], Sequence[str]]:
        """Translate `eqn` into its SDFG equivalent.

        To do this the function will do the following steps:
        - Assemble the in and output variables.
        - Select which subtranslator to use.
        - Create a new empty state, i.e. append to the tentative terminal state.
        - Perform the actual translation.

        Returns:
            The SDFG names that where used as input and output are returned.
            The inputs might contain `None` which indicates that the input was a Jax literal.
            For more information see `JaCeSubTranslatorInterface.can_translate_jaxeqn()`.

        Notes:
            While `jaxpr` must be the closed version, `eqn` must come from the unclosed version.
            The function will also perform some consistency checking.
        """
        assert isinstance(eqn, jcore.JaxprEqn)
        assert isinstance(jaxpr, jcore.ClosedJaxpr)

        if len(eqn.effects) != 0:
            raise NotImplementedError(f"Equation '{eqn}' had side effects.")

        # Input/Output variables
        in_var_names: Sequence[str | None] = self._create_jax_var_list(
            eqn.invars,
            prevent_creation=True,  # Inputs must already exists.
        )
        out_var_names: Sequence[str] = self._create_jax_var_list(  # type: ignore[assignment]
            eqn.outvars,
            only_creation=True,  # Output must not exist yet.
        )

        # Find the subtranslator
        subtranslator: translator.JaCeSubTranslatorInterface = self._find_sub_translator_for(
            in_var_names=in_var_names,
            out_var_names=out_var_names,
            eqn=eqn,
        )

        # Create the state into which the equation is put
        last_term_state: dace.SDFGState = self.get_terminal_sdfg_state()  # noqa: F841 # Will be used later
        eqn_state = self.append_new_state(
            label=f"{eqn.primitive.name}_{out_var_names[0]}",
            prev_state=None,  # Force to append as terminal state.
        )

        # Now perform the actual translation of the equation.
        new_sdfg_term_state = subtranslator.translate_jaxeqn(
            driver=self,
            in_var_names=in_var_names,
            out_var_names=out_var_names,  # Might be modified by subtranslator!
            eqn=eqn,
            eqn_state=eqn_state,
        )

        # Determine the new (tentative) terminal state of the SDFG we are building.
        if new_sdfg_term_state is None:
            if eqn_state is self._term_sdfg_state:
                raise RuntimeError("Inconsistent terminal state was detected.")
            new_sdfg_term_state = eqn_state
        elif isinstance(new_sdfg_term_state, dace.SDFGState):
            # TODO(phimuell): use `last_term_state` to test if there is reachability to new end.
            pass
        else:
            raise TypeError(f"Encountered illegal types '{type(new_sdfg_term_state)}'")

        # In case a subtranslator decided to not use the variables we created for it, he is technically
        #  allowed to create new ones, but he must update the `out_var_names` list.
        #  We will now test if the mapping was updated correctly.
        for expectedSDFGName, jax_var in zip(out_var_names, eqn.outvars, strict=False):
            mapped_sdfg_name = self.map_jax_var_to_sdfg(jax_var)
            jax_name = jutil.get_jax_var_name(jax_var)
            if mapped_sdfg_name != expectedSDFGName:
                raise ValueError(
                    f"Mapping inconsistency detected, expected that Jax variable '{jax_name}' maps to '{expectedSDFGName}' but it actually maps to '{mapped_sdfg_name}'."
                )

        # Views can only be used if there is a direct connection, between source, view and destination (place of usage)
        #  Because of the way how Jax works, it is impossible that an output variable is a View.
        #  Thus we now make the check if this is the case.
        for outVarName, jax_var in zip(out_var_names, eqn.outvars, strict=False):
            sdfg_var = self.get_array(outVarName)
            if isinstance(sdfg_var, (dace.data.Array, dace.data.Scalar)):
                pass
            elif isinstance(sdfg_var, dace.data.View):
                raise TypeError(
                    f"For the Jax variable '{jutil.get_jax_var_name(jax_var)}' (SDFG: '{outVarName}'), which is an output, you used a View, which is not possible."
                    + " It must either be an array or a scalar."
                )
            else:
                raise NotImplementedError(
                    f"The output variable '{jutil.get_jax_var_name(jax_var)}' (SDFG: '{outVarName}') is of type '{type(sdfg_var).__name__}' which I does not know how to handle."
                )

        # Modify terminal head state of 'self'
        self._term_sdfg_state = new_sdfg_term_state

        return (in_var_names, out_var_names)

    def _translate_jaxpr_internal(
        self,
        jaxpr: jcore.ClosedJaxpr,
    ) -> jtrutil.JaCeTranslationMemento:
        """Performs the actual translation of the Jaxpr into an SDFG.

        The function assumes that the context is already allocated and the initial variables are already created.
        The function will ignore, i.e. not translate, any state whose output variables name only consists of `_`.

        The function will store the internal state of `self` into a memento and return it.
        However, it will not deallocate the context of `self`, thus `self` and the memento share the same context in memory.

        Args:
            jaxpr:      The Jaxpr to translate.

        Notes:
            The function will unconditionally handle empty Jaxpr.
            Jax uses a variable with name `_` to indicate that this value is never read.
                It is included by some transformations such as `grad()`.
        """
        assert isinstance(jaxpr, jcore.ClosedJaxpr)
        assert self.is_allocated()

        nb_translated_eqn: int = 0
        for eqn in jaxpr.jaxpr.eqns:  # Translate the equations one by one.
            assert len(eqn.effects) == 0
            if len(eqn.outvars) == 0:  # Do we need this special case.
                continue  #  Looks more like internal Jax error.
            if any(jutil.get_jax_var_name(outVar) == "_" for outVar in eqn.outvars):
                assert (len(eqn.outvars) == 1) or all(
                    jutil.get_jax_var_name(outVar) == "_" for outVar in eqn.outvars
                )
                continue
            _, out_var_names = self._translate_single_eqn(jaxpr=jaxpr, eqn=eqn)
            nb_translated_eqn += 1

        if nb_translated_eqn != 0:
            # Equations where translated so set the output variables.
            self._sdfg_out_names = tuple(out_var_names)
        else:
            # No equations were translated, i.e. no equation at all or all outputs had name '_'
            self._handle_null_jaxpr(jaxpr)

        return self._export_memento()

    def _export_memento(self) -> jtrutil.JaCeTranslationMemento:
        """Encapsulate the translation context of `self` into a memento.

        This function will not deallocate the internal context of `self`.
        Thus the memento and `self` share the same context in memory.
        """
        assert self.is_allocated()
        assert len(self._sdfg_in_names) > 0
        assert all(isinstance(x, str) for x in self._sdfg_in_names)
        assert len(self._sdfg_out_names) > 0
        assert all(isinstance(x, str) for x in self._sdfg_out_names)

        return jtrutil.JaCeTranslationMemento(
            sdfg=self._sdfg,
            start_state=self._init_sdfg_state,
            terminal_state=self._term_sdfg_state,
            jax_name_map=self._jax_name_map,
            inp_names=self._sdfg_in_names,
            out_names=self._sdfg_out_names,
        )

    def _handle_null_jaxpr(
        self,
        jaxpr: jcore.ClosedJaxpr,
    ) -> JaxprTranslationDriver:
        """This function is called in case a `Jaxpr` with zero equations is encountered.

        Notes:
            This function will fill the internal list of outputs.
        """
        if len(jaxpr.eqns) != 0:
            raise NotImplementedError("'_handle_null_jaxpr()' was called for a non empty Jaxpr.")
        if (
            len(jaxpr.out_avals) == 0
        ):  # There is not output so we do not have to copy anything around.
            self._sdfg_out_names = ()
            return self
        if self.is_head_translator():
            # In this case there is nothing to do, because input is already the output.
            #  However, this is only possible if we are the head translator.
            self._sdfg_out_names = tuple(
                self.map_jax_var_to_sdfg(jax_out_var) for jax_out_var in jaxpr.jaxpr.outvars
            )
            raise NotImplementedError("Please test me.")
            return self  # type: ignore[unreachable] # reminder
        #
        assert self._term_sdfg_state is self._init_sdfg_state
        assert len(self._sdfg_in_names) > 0
        assert len(self._sdfg_out_names) == 0

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
            sdfg_out_name = self._add_array(
                jax_out_var,
                as_transient=True,
                name_prefix="_zero_equation_output_for_",
                update_var_mapping=False,
            )

            # We now create a new mapping, we do this that we will later find the variable again.
            self._add_jax_name_mapping(jax_var=jax_out_name, sdfg_name=sdfg_out_name)
            out_var_names.append(jax_out_name)

            # Now copy the input into the fake output variable.
            inp_acc = self._init_sdfg_state.add_read(self.map_jax_var_to_sdfg(jax_inp_name))
            out_acc = self._init_sdfg_state.add_write(self.map_jax_var_to_sdfg(jax_out_var))
            self._init_sdfg_state.add_nedge(
                src=inp_acc,
                dst=out_acc,
                data=dace.Memlet.from_array(
                    jax_inp_name, self.get_array(self.map_jax_var_to_sdfg(jax_inp_name))
                ),
            )
        # We also have to update the list of outputs.
        #  This is needed for making the exporter aware of what we are doing.
        self._sdfg_out_names = tuple(out_var_names)
        return self

    # fmt: off
    _forbidden_names: Final[set[str]] = {
        # These should be most of the C++ keywords, it is more important to have the short ones.
        #  Taken from 'https://learn.microsoft.com/en-us/cpp/cpp/keywords-cpp?view=msvc-170'
        'alignas', 'alignof', 'and', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch',
        'char', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit', 'continue',
        'decltype', 'default', 'delete', 'directive', 'do', 'double', 'else', 'enum', 'explicit', 'export',
        'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable',
        'namespace', 'new', 'noexcept', 'not', 'nullptr', 'operator', 'or', 'private', 'protected',
        'public', 'register', 'requires', 'return', 'short', 'signed', 'sizeof', 'static', 'struct',
        'switch', 'template', 'this', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union',
        'unsigned', 'using', 'virtual', 'void', 'volatile', 'while', 'xor', 'std',
    }
    # fmt: on
