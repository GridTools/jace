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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import dace
import jax
import jax.core as jcore


@dataclass(init=True, repr=True, eq=True, frozen=True, slots=True)
class JaCeVar:
    """Substitute class for Jax' `Var` instance.

    This class is similar to a `jax.core.Var` class, but much simpler.
    It is only a container for a name, shape and a datatype.
    All extractor functions `get_jax_var{name, shape, dtype}()` will accept it,
    as well as multiple functions of the driver.

    Notes:
        Main intention is to test functionality.
    """

    name: str
    shape: tuple[int | dace.symbol | str, ...] | int | dace.symbol | str | tuple[()]
    dtype: dace.typeclass


def get_jax_var_name(jax_var: jcore.Atom | JaCeVar | str) -> str:
    """Returns the name of the Jax variable as a string.

    Args:
        jax_var:     The variable to stringify.
    """
    if isinstance(jax_var, jcore.DropVar):
        return "_"
    if isinstance(jax_var, JaCeVar):
        jax_name = jax_var.name
    elif isinstance(jax_var, jcore.Var):
        # This stopped working after version 0.20.4, because of some changes in Jax
        #  See `https://github.com/google/jax/pull/10573` for more information.
        #  The following implementation will generate stable names, but decouples
        #  them from pretty printed Jaxpr, we maybe need a pretty print context somewhere.
        jax_name = f"jax{jax_var.count}{jax_var.suffix}"
    elif isinstance(jax_var, jcore.Literal):
        raise TypeError("Can not translate a Jax Literal to a variable name.")
    elif isinstance(jax_var, str):
        jax_name = jax_var
    else:
        raise TypeError(
            f"Can not transform '{jax_var}' (type: '{type(jax_var).__name__}') not a name."
        )
    assert isinstance(jax_name, str)
    if not (
        re.fullmatch("jax[1-9][0-9]*", jax_name) or re.fullmatch("[a-zA-Z][a-zA-Z]*", jax_name)
    ):
        raise ValueError(f"Deduced Jax name '{jax_name}' is invalid.")
    return jax_name


def get_jax_var_shape(jax_var: jcore.Atom) -> tuple[int, ...]:
    """Returns the shape of a Jax variable.

    Args:
        jax_var:     The variable to process
    """
    if isinstance(jax_var, jcore.Atom):
        return jax_var.aval.shape
    if isinstance(jax_var, JaCeVar):
        assert isinstance(jax_var.shape, tuple)
        return jax_var.shape
    raise TypeError(f"'get_jax_var_shape()` is not implemented for '{type(jax_var)}'.")


def get_jax_var_dtype(jax_var: jcore.Atom) -> dace.typeclass:
    """Returns the DaCe equivalent of `jax_var`s datatype."""
    if isinstance(jax_var, jcore.Atom):
        return translate_dtype(jax_var.aval.dtype)
    if isinstance(jax_var, JaCeVar):
        return translate_dtype(jax_var.dtype)
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
    from jax._src import pjit as jaxpjit
    import jaxlib

    # These are all types we consider as jaxify
    jaxify_types: Sequence[type] = (
        jcore.Primitive,
        # jstage.Wrapped, # Not runtime chakable
        jaxpjit.JitWrapped,
        jaxlib.xla_extension.PjitFunction
    )
    return isinstance(obj, jaxify_types)


def translate_dtype(dtype: Any) -> dace.typeclass:
    """Turns a Jax datatype into a DaCe datatype."""

    if isinstance(dtype, dace.typeclass):
        return dtype

    # Make some basic checks if the datatype is okay
    name_of_dtype = str(dtype)
    if (not jax.config.read("jax_enable_x64")) and (name_of_dtype == "float64"):
        raise ValueError("Found a 'float64' type but 'x64' support is disabled.")
    if name_of_dtype.startswith("complex"):
        raise NotImplementedError("Support for complecx computation is not implemented yet.")

    # Now extract the datatype from dace, this is extremely ugly.
    if not hasattr(dace.dtypes, name_of_dtype):
        raise TypeError(f"Could not find '{name_of_dtype}' ({type(dtype).__name__}) in 'dace'.")
    dcd_type = getattr(dace.dtypes, name_of_dtype)

    if not isinstance(dcd_type, dace.dtypes.typeclass):
        raise TypeError(
            f"'{name_of_dtype}' does not map to a 'dace.typeclass' but to a '{type(dcd_type).__name__}'."
        )
    return dcd_type
