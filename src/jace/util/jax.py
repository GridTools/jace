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

    Todos:
        Implement a regex check for the name.
    """
    if isinstance(jax_var, jcore.DropVar):
        return "_"
    if isinstance(jax_var, JaCeVar):
        return jax_var.name
    if isinstance(jax_var, jcore.Atom):
        jax_name = str(jax_var)  # This only works up to some version
    elif isinstance(jax_var, str):
        jax_name = jax_var
    else:
        raise TypeError(
            f"Does not know how to transform '{jax_var}' (type: '{type(jax_var).__name__}') into a string."
        )
    # TODO(phimuell): Add regex to ensure that the name is legit.
    assert isinstance(jax_name, str)
    if len(jax_name) == 0:
        raise ValueError(
            f"Failed to translate the Jax variable '{jax_var}' into a name, the result was empty."
        )
    return jax_var


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
