# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements all utility functions that are related to Jax.

Most of the functions defined here allow an unified access to Jax' internals
in a consistent and stable way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import dace
import jax.core as jcore


# Used by `get_jax_var_name()` to test if a name for a jax variable is valid.
_VALID_JAX_NAME_PATTERN: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")


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
    match jax_var:
        case jcore.DropVar():
            return "_"

        case JaCeVar():
            jax_name = jax_var.name

        case jcore.Var():
            # This stopped working after version 0.20.4, because of some changes in Jax
            #  See `https://github.com/google/jax/pull/10573` for more information.
            #  The following implementation will generate stable names, but decouples
            #  them from pretty printed Jaxpr, we maybe need a pretty print context somewhere.
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


def translate_dtype(dtype: Any) -> dace.typeclass:
    """Turns a Jax datatype into a DaCe datatype."""
    if isinstance(dtype, dace.typeclass):
        return dtype
    if dtype is None:
        # Special behaviour of `dtype_to_typeclass()`
        raise NotImplementedError()
    return dace.dtype_to_typeclass(dtype)
