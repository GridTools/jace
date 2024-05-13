# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements all utility functions that are related to Jax.

Most of the functions defined here allow an unified access to Jax' internals
in a consistent and stable way.
It is important that this module is different from the `jace.jax` module, which
mimics the full `jax` package itself.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, overload

import dace
import jax.core as jax_core
import jax.dtypes as jax_dtypes
import numpy as np

from jace import util
from jace.util import util as dcutil  # Partially initialized module


@dcutil.dataclass_with_default_init(init=True, repr=True, frozen=True, slots=True)
class JaCeVar:
    """Substitute class for Jax' `Var` instance.

    This class can be seen as some kind of substitute `jax.core.Var`.
    The main intention of this class is as an internal representation of values,
    as they are used in Jax, but without the Jax machinery.
    The main differences to Jax variable is, that this class has a name and also a storage type.

    Notes:
        Main intention is to test functionality.
        If the name of a `JaCeVar` is '_' it is considered a drop variable.
        If the name of a `JaCeVar` is empty, the automatic naming will consider it as a Jax variable.
        The definition of `__hash__` and `__eq__` is in accordance how Jax variable works.

    Todo:
        Do we need strides for caching; I would say so.
    """

    name: str
    shape: tuple[int | dace.symbol | str, ...] | tuple[()]
    dtype: dace.typeclass
    storage: dace.StorageType = dace.StorageType.Default

    def __init__(
        self,
        name: str,
        shape: Sequence[int | dace.symbol | str] | int | dace.symbol | str,
        dtype: Any,
        storage: dace.StorageType = dace.StorageType.Default,
    ) -> None:
        if name == "":
            pass  # Explicit allowed in the interface, but a bit strange.
        elif (name != "_") and (not util.VALID_SDFG_VAR_NAME.fullmatch(name)):
            raise ValueError(f"Passed an invalid name '{name}'.")
        if isinstance(shape, (int, dace.symbol, str)):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            shape = tuple(shape)
        if not isinstance(dtype, dace.typeclass):
            dtype = translate_dtype(dtype)
        assert all(isinstance(x, (int, dace.symbol, str)) for x in shape)
        assert isinstance(storage, dace.StorageType)
        self.__default_init__(name=name, shape=shape, dtype=dtype, storage=storage)  # type: ignore[attr-defined]  # __default_init__ is existing.

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, JaCeVar):
            return NotImplemented
        return id(self) == id(other)


def get_jax_var_name(jax_var: jax_core.Atom | JaCeVar | str) -> str:
    """Returns the name of the Jax variable as a string.

    Args:
        jax_var:     The variable to stringify.

    Notes:
        Due to some modification in Jax itself, this function is unable to return "proper" variable names.
        This function is subject for removal.
    """
    match jax_var:
        case jax_core.DropVar():
            return "_"
        case JaCeVar():
            # In case of an empty name consider the jace variable as a Jax variable.
            #  This is mostly for testing.
            jax_name = f"jax{id(jax_var)}" if jax_var.name == "" else jax_var.name
        case jax_core.Var():
            # This stopped working after version 0.20.4, because of some changes in Jax
            #  See `https://github.com/google/jax/pull/10573` for more information.
            #  The following implementation will generate stable names, however, they will be decoupled
            #  from output of the pretty printed Jaxpr
            jax_name = f"jax{jax_var.count}{jax_var.suffix}"
        case jax_core.Literal():
            raise TypeError("Can not derive a name from a Jax Literal.")
        case str():
            jax_name = jax_var
        case _:
            raise TypeError(
                f"Does not know how to transform '{jax_var}' (type: '{type(jax_var).__name__}') into a string."
            )
    assert isinstance(jax_name, str)

    if not util.VALID_JAX_VAR_NAME.fullmatch(jax_name):
        raise ValueError(f"Deduced Jax name '{jax_name}' is invalid.")
    return jax_name


@overload
def get_jax_var_shape(jax_var: JaCeVar) -> tuple[int | dace.symbol | str, ...] | tuple[()]: ...  # type: ignore[overload-overlap]


@overload
def get_jax_var_shape(jax_var: jax_core.Atom) -> tuple[int, ...] | tuple[()]: ...


def get_jax_var_shape(
    jax_var: jax_core.Atom | JaCeVar,
) -> tuple[int | dace.symbol | str, ...] | tuple[()]:
    """Returns the shape of a Jax variable.

    Args:
        jax_var:     The variable to process
    """
    match jax_var:
        case jax_core.Var() | jax_core.Literal():
            return jax_var.aval.shape
        case JaCeVar():
            return jax_var.shape
        case _:
            raise TypeError(f"'get_jax_var_shape()` is not implemented for '{type(jax_var)}'.")


def get_jax_var_dtype(jax_var: jax_core.Atom | JaCeVar) -> dace.typeclass:
    """Returns the DaCe equivalent of `jax_var`s datatype."""
    match jax_var:
        case jax_core.Var() | jax_core.Literal():
            return translate_dtype(jax_var.aval.dtype)
        case JaCeVar():
            return translate_dtype(jax_var.dtype)
        case _:
            raise TypeError(f"'get_jax_var_dtype()` is not implemented for '{type(jax_var)}'.")


def is_tracing_ongoing(
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Test if tracing is ongoing.

    While a return value `True` guarantees that a translation is ongoing,
    a value of `False` does not guarantees that no tracing is active.

    Raises:
        RuntimeError: If the function fails to make a detection.
    """
    # The current implementation only checks the arguments if it contains tracers.
    if (len(args) == 0) and (len(kwargs) == 0):
        raise RuntimeError("Failed to determine if tracing is ongoing.")
    return any(isinstance(x, jax_core.Tracer) for x in itertools.chain(args, kwargs.values()))


def translate_dtype(dtype: Any) -> dace.typeclass:
    """Turns a Jax datatype into a DaCe datatype."""
    if isinstance(dtype, dace.typeclass):
        return dtype
    if dtype is None:
        # Special behaviour of `dtype_to_typeclass()`
        raise NotImplementedError()

    # For reasons unknown to me we have to do the dtype conversion this way.
    #  It is not possible to simply call `dace.typeclass(dtype)` or pass it to
    #  `dace.dtype_to_typeclass()`, it will generate an error.
    #  We keep the `dtype_to_typeclass()` function call, in order to handle
    #  NumPy types as DaCe intended them to be handled.
    try:
        return dace.dtype_to_typeclass(dtype)
    except KeyError:
        pass

    try:
        dtype_ = jax_dtypes.canonicalize_dtype(dtype)
        return dace.dtype_to_typeclass(dtype_)
    except Exception:
        pass

    dtype_name = str(dtype)
    if hasattr(dace.dtypes, dtype_name):
        return getattr(dace.dtypes, dtype_name)
    if hasattr(np, dtype_name):
        dtype = getattr(np, dtype)
        return dace.dtype_to_typeclass(dtype)
    raise ValueError(f"Unable to translate '{dtype}' ino a DaCe dtype.")


def _propose_jax_name(
    jax_var: jax_core.Atom | JaCeVar,
    jax_name_map: Mapping[jax_core.Var | JaCeVar, Any] | None = None,
) -> str:
    """Proposes a variable name for `jax_var`.

    There are two modes for proposing new names.
    In the first mode, `get_jax_var_name()` is used to derive a name.
    The second mode, proposes a name based on all names that are already known,
    this leads to names similar to the ones used by Jax.

    Args:
        jax_var:        The variable for which a name to propose.
        jax_name_map:   A mapping of all Jax variables that were already named.

    Notes:
        The second mode is activated by passing `jax_name_map` as argument.
        The naming of variables are only consistent with the inner most Jaxpr a variable is defined in.
        Dropped variables will always be named `'_'`.
    """
    if util.traits.is_drop_var(jax_var):
        return "_"
    if isinstance(jax_var, jax_core.Literal):
        raise TypeError(f"Can not propose a name for literal '{jax_var}'.")
    if jax_name_map is None:
        return get_jax_var_name(jax_var)
    if jax_var in jax_name_map:
        # Should be turned into a lookup?
        raise RuntimeError(
            f"Can not propose a second name for '{jax_var}', it already known as '{jax_name_map[jax_var]}'."
        )
    if isinstance(jax_var, jax_core.Var):
        pass
    elif isinstance(jax_var, JaCeVar):
        # If the name of the JaCe variable is empty, then use the name proposing
        #  technique used for Jax variables; Mostly used for debugging.
        if jax_var.name != "":
            return jax_var.name
    else:
        raise TypeError(f"Can not propose a name for '{jax_var}'")

    c = len(jax_name_map)
    jax_name = ""
    while len(jax_name) == 0 or c != 0:
        c, i = c // 26, c % 26
        jax_name = chr(97 + i % 26) + jax_name
    return jax_name + getattr(jax_var, "suffix", "")
