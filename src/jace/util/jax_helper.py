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

import dataclasses
import itertools
from collections.abc import Mapping
from typing import Any

import dace
import jax.core as jax_core

import jace.util as util


@dataclasses.dataclass(repr=True, frozen=True, eq=False)
class JaCeVar:
    """Replacement for the `jax.Var` class.

    This class can be seen as some kind of substitute `jax.core.Var`.
    The main intention of this class is as an internal representation of values, as they are used in Jax, but without the Jax machinery.
    As abstract values in Jax this class has a datatype, which is a `dace.typeclass` instance and a shape.
    In addition it has an optional name, which allows to create variables with a certain name using `JaxprTranslationDriver.add_array()`.

    Note:
        If the name of a `JaCeVar` is '_' it is considered a drop variable.
        The definitions of `__hash__` and `__eq__` are in accordance how Jax variable works.

    Todo:
        - Add support for strides.
    """

    shape: tuple[int | dace.symbol | str, ...]
    dtype: dace.typeclass
    name: str | None = None

    def __post_init__(self) -> None:
        """Sanity checks."""
        if self.name is not None and (
            (not util.VALID_SDFG_VAR_NAME.fullmatch(self.name))
            or self.name in util.FORBIDDEN_SDFG_VAR_NAMES
        ):
            raise ValueError(f"Supplied the invalid name '{self.name}'.")
        if not isinstance(self.dtype, dace.typeclass):  # No typechecking yet.
            raise TypeError(f"'dtype' is not a 'dace.typeclass' but '{type(self.dtype).__name__}'.")

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, JaCeVar):
            return NotImplemented
        return id(self) == id(other)


def get_jax_var_name(jax_var: jax_core.Atom | JaCeVar) -> str:
    """Returns the name of the Jax variable as a string.

    Args:
        jax_var:     The variable to stringify.

    Notes:
        If `jax_var` is a `JaCeVar` the function will return, if defined, its `.name` property.
        Otherwise it will compose a name similar to Jax `Var` objects.
        The returned names are stable, i.e. it will output the same value for the same variable.
    """
    match jax_var:
        case jax_core.DropVar():
            return "_"
        case JaCeVar():
            return jax_var.name if jax_var.name else f"jax{id(jax_var)}"
        case jax_core.Var():
            # This is not how the pretty printer works nor Jax.Var.__repr__, but leads to stable names that can be used.
            return f"jax{jax_var.count}{jax_var.suffix}"
        case jax_core.Literal():
            raise TypeError("Can not derive a name from a Jax Literal.")
        case _:
            raise TypeError(
                f"Does not know how to transform '{jax_var}' (type: '{type(jax_var).__name__}') into a string."
            )


def get_jax_var_shape(jax_var: jax_core.Atom | JaCeVar) -> tuple[int | dace.symbol | str, ...]:
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
    if dtype is None:
        raise NotImplementedError  # Handling a special case in DaCe.
    if isinstance(dtype, dace.typeclass):
        return dtype
    try:
        return dace.typeclass(dtype)
    except (NameError, KeyError):
        pass
    return dace.dtype_to_typeclass(getattr(dtype, "type", dtype))


def propose_jax_name(
    jax_var: jax_core.Atom | JaCeVar,
    jax_name_map: Mapping[jax_core.Var | JaCeVar, str] | None = None,
) -> str:
    """Proposes a variable name for `jax_var`.

    If `jax_name_map` is `None` then the function will fallback to `get_jax_var_name()`.
    If `jax_name_map` is supplied the function will:
    - if `jax_var` is stored inside `jax_name_map` this value will be returned.
    - if `jax_var` is a `JaCeVar` with a set `.name` property it will be returned.
    - otherwise the function will generate a new name similar to how the pretty printer of Jaxpr works.

    Args:
        jax_var:        The variable for which a name to propose.
        jax_name_map:   A mapping of all Jax variables that were already named.

    Note:
        The function guarantees that the returned name passes `VALID_SDFG_VAR_NAME` test
        and that the name is not part of `util.FORBIDDEN_SDFG_VAR_NAMES`.
        Dropped variables will always be named `'_'`.
    """
    if isinstance(jax_var, jax_core.Literal):
        raise TypeError(f"Can not propose a name for literal '{jax_var}'.")
    if util.is_drop_var(jax_var) or (jax_name_map is None):
        return get_jax_var_name(jax_var)
    if jax_var in jax_name_map:
        return jax_name_map[jax_var]
    if isinstance(jax_var, JaCeVar) and (jax_var.name is not None):
        return jax_var.name

    # We have the set of all previous names, so we generate names
    #  in the same way as Jax does:
    c = len(jax_name_map)
    jax_name = ""
    while len(jax_name) == 0 or c != 0:
        c, i = c // 26, c % 26
        jax_name = chr(97 + i) + jax_name
    jax_name = jax_name + getattr(jax_var, "suffix", "")

    if jax_name in util.FORBIDDEN_SDFG_VAR_NAMES:
        jax_name = f"__jace_forbidden_{jax_name}"
    return jax_name
