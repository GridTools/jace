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

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import dace
import jax.core as jcore
import numpy as np


# Used by `get_jax_var_name()` to test if a name for a jax variable is valid.
_VALID_JAX_NAME_PATTERN: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")


@dataclass(init=True, repr=True, frozen=True, slots=True)
class JaCeVar:
    """Substitute class for Jax' `Var` instance.

    This class is similar to a `jax.core.Var` class, but much simpler.
    It is only a container for a name, shape and a datatype.
    All extractor functions `get_jax_var{name, shape, dtype}()` will accept it, as well as multiple functions of the driver.

    Notes:
        Main intention is to test functionality.
        While for a Jax `Var` object the name is rather irrelevant, `JaCeVar` use their name.
    """

    name: str
    shape: tuple[int | dace.symbol | str, ...] | int | dace.symbol | str | tuple[()]
    dtype: dace.typeclass

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(
        self,
        other: Any
    ) -> bool:
        if(not isinstance(other, JaCeVar)):
            return NotImplemented
        return self.name == other.name



def get_jax_var_name(jax_var: jcore.Atom | JaCeVar | str) -> str:
    """Returns the name of the Jax variable as a string.

    Args:
        jax_var:     The variable to stringify.

    Notes:
        Due to some modification in Jax itself, this function is unable to return "proper" variable names.
        This function is subject for removal.
    """
    match jax_var:
        case jcore.DropVar():
            return "_"

        case JaCeVar():
            jax_name = jax_var.name

        case jcore.Var():
            # This stopped working after version 0.20.4, because of some changes in Jax
            #  See `https://github.com/google/jax/pull/10573` for more information.
            #  The following implementation will generate stable names, however, they will be decoupled
            #  from output of the pretty printed Jaxpr
            jax_name = f"jax{jax_var.count}{jax_var.suffix}"

        case jcore.Literal():
            raise TypeError("Can not derive a name from a Jax Literal.")

        case str():
            jax_name = jax_var

        case _:
            raise TypeError(
                f"Does not know how to transform '{jax_var}' (type: '{type(jax_var).__name__}') into a string."
            )
    assert isinstance(jax_name, str)

    if not _VALID_JAX_NAME_PATTERN.fullmatch(jax_name):
        raise ValueError(f"Deduced Jax name '{jax_name}' is invalid.")
    return jax_name


def get_jax_var_shape(jax_var: jcore.Atom | JaCeVar) -> tuple[int, ...]:
    """Returns the shape of a Jax variable.

    Args:
        jax_var:     The variable to process
    """
    match jax_var:
        case jcore.Var() | jcore.Literal():
            return jax_var.aval.shape

        case JaCeVar():
            return jax_var.shape

        case _:
            raise TypeError(f"'get_jax_var_shape()` is not implemented for '{type(jax_var)}'.")


def get_jax_var_dtype(jax_var: jcore.Atom | JaCeVar) -> dace.typeclass:
    """Returns the DaCe equivalent of `jax_var`s datatype."""
    match jax_var:
        case jcore.Var() | jcore.Literal():
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
    from itertools import chain

    # The current implementation only checks the arguments if it contains tracers.
    if (len(args) == 0) and (len(kwargs) == 0):
        raise RuntimeError("Failed to determine if tracing is ongoing.")
    return any(isinstance(x, jcore.Tracer) for x in chain(args, kwargs.values()))


def is_jaxified(obj: Any) -> bool:
    """Tests if `obj` is a "jaxified" object.

    A "jexified" object is an object that was processed by Jax.
    While a return value of `True` guarantees a jaxified object,
    `False` might not proof the contrary.
    """
    import jaxlib
    from jax._src import pjit as jaxpjit

    # These are all types we consider as jaxify
    return isinstance(
        obj,
        (
            jcore.Primitive,
            # jstage.Wrapped, # Not runtime chakable
            jaxpjit.JitWrapped,
            jaxlib.xla_extension.PjitFunction,
        ),
    )


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
        dtype_name = str(dtype)
        if hasattr(dace.dtypes, dtype_name):
            return getattr(dace.dtypes, dtype_name)
        if hasattr(np, dtype_name):
            dtype = getattr(np, dtype)
            return dace.dtype_to_typeclass(dtype)


def is_drop_var(jax_var: jcore.Atom | JaCeVar) -> bool:
    """Tests if `jax_var` is a drop variable.
    """

    if(isinstance(jax_var, jcore.DropVar)):
        return True
    if(isinstance(jax_var, JaCeVar)):
        return jax_var.name == '_'
    return False


def _propose_jax_name(
        jax_var: jcore.Atom | JaCeVar,
        jax_name_map: Optional[Mapping[jcore.Var | JaCeVar, Any]] = None,
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
    if(is_drop_var(jax_var)):
        return "_"
    if(isinstance(jax_var, jcore.Literal)):
        raise TypeError(f"Can not propose a name for literal '{jax_var}'.")
    if(jax_var in jax_name_map):
        raise RuntimeError(
            f"Can not propose a second name for '{jax_var}', it already known as '{jax_name_map[jax_var]}'."
        )
    if(jax_name_map is None):
        return get_jax_var_name(jax_var)
    if(isinstance(jax_var, JaCeVar)):
        return jax_var.name
    assert isinstance(jax_var, jcore.Atom)

    c = len(jax_name_map)
    jax_name = ""
    while len(jax_name) == 0 or c == 0:
        c, i = c // 26, c % 26
        jax_name = chr(97 + i % 26) + jax_name
    return jax_name





