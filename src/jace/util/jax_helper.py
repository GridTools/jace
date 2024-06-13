# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Implements all utility functions that are related to Jax.

Most of the functions defined here allow an unified access to Jax' internal in
a consistent and stable way.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING, Any

import dace
import jax
import jax.core as jax_core
import numpy as np

from jace import util


if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclasses.dataclass(repr=True, frozen=True, eq=False)
class JaCeVar:
    """
    Replacement for the `jax.Var` class.

    This class can be seen as some kind of substitute `jax.core.Var`. The main
    intention of this class is as an internal representation of values, as they
    are used in Jax, but without the Jax machinery. As abstract values in Jax
    this class has a datatype, which is a `dace.typeclass` instance and a shape.
    In addition it has an optional name, which allows to create variables with
    a certain name using `JaxprTranslationBuilder.add_array()`.

    If it is expected that code must handle both Jax variables and `JaCeVar`
    then the `get_jax_var_*()` functions should be used.

    Args:
        shape: The shape of the variable.
        dtype: The dace datatype of the variable.
        name: Name the variable should have, optional.

    Note:
        If the name of a `JaCeVar` is '_' it is considered a drop variable. The
        definitions of `__hash__` and `__eq__` are in accordance with how Jax
        variable works.

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
    """Returns the name of `jax_var` as a string."""
    match jax_var:
        case jax_core.DropVar():
            return "_"
        case JaCeVar():
            return jax_var.name if jax_var.name else f"jax{id(jax_var)}"
        case jax_core.Var():
            # This is not how the pretty printer works nor `jax.Var.__repr__()`,
            #  but leads to stable and valid names.
            return f"jax{jax_var.count}{jax_var.suffix}"
        case jax_core.Literal():
            raise TypeError("Can not derive a name from a Jax Literal.")
        case _:
            raise TypeError(
                f"Does not know how to transform '{jax_var}' (type: '{type(jax_var).__name__}') "
                "into a string."
            )


def get_jax_var_shape(jax_var: jax_core.Atom | JaCeVar) -> tuple[int | dace.symbol | str, ...]:
    """Returns the shape of `jax_var`."""
    match jax_var:
        case jax_core.Var() | jax_core.Literal():
            assert hasattr(jax_var.aval, "shape")  # To silences mypy.
            return jax_var.aval.shape
        case JaCeVar():
            return jax_var.shape
        case _:
            raise TypeError(f"'get_jax_var_shape()` is not implemented for '{type(jax_var)}'.")


def get_jax_var_dtype(jax_var: jax_core.Atom | JaCeVar) -> dace.typeclass:
    """Returns the DaCe equivalent of `jax_var`s datatype."""
    match jax_var:
        case jax_core.Var() | jax_core.Literal():
            assert hasattr(jax_var.aval, "dtype")  # To silences mypy.
            return translate_dtype(jax_var.aval.dtype)
        case JaCeVar():
            return jax_var.dtype
        case _:
            raise TypeError(f"'get_jax_var_dtype()` is not implemented for '{type(jax_var)}'.")


def is_tracing_ongoing(*args: Any, **kwargs: Any) -> bool:
    """
    Test if tracing is ongoing.

    While a return value `True` guarantees that a translation is ongoing, a
    value of `False` does not guarantees that no tracing is ongoing.
    """
    # To detect if there is tracing ongoing, we check the internal tracing stack of Jax.
    #  Note that this is highly internal and depends on the precise implementation of
    #  Jax. For that reason we first look at all arguments and check if they are
    #  tracers. Furthermore, it seems that Jax always have a bottom interpreter on the
    #  stack, thus it is empty if `len(...) == 1`!
    #  See also: https://github.com/google/jax/pull/3370
    if any(isinstance(x, jax_core.Tracer) for x in itertools.chain(args, kwargs.values())):
        return True
    if len(jax._src.core.thread_local_state.trace_state.trace_stack.stack) == 1:
        return False
    if len(jax._src.core.thread_local_state.trace_state.trace_stack.stack) > 1:
        return True
    raise RuntimeError("Failed to determine if tracing is ongoing.")


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
    """
    Proposes a variable name for `jax_var`.

    If `jax_name_map` is `None` the function will fallback to
    `get_jax_var_name(jax_var)`. If `jax_name_map` is supplied the function
    will:
    - If `jax_var` is stored inside `jax_name_map`, returns the mapped value.
    - If `jax_var` is a `JaCeVar` with a set `.name` property that name will
        be returned.
    - Otherwise the function will generate a new name in a similar way to the
        pretty printer of Jaxpr.

    Args:
        jax_var: The variable for which a name to propose.
        jax_name_map: A mapping of all Jax variables that were already named.

    Note:
        The function guarantees that the returned name passes `VALID_SDFG_VAR_NAME`
        test and that the name is not inside `util.FORBIDDEN_SDFG_VAR_NAMES`.
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

    # This code is taken from Jax so it will generate similar ways, the difference is
    #  that we do the counting differently.
    #  Note that `z` is followed by `ba` and not `aa` as it is in Excel.
    c = len(jax_name_map)
    jax_name = ""
    while len(jax_name) == 0 or c != 0:
        c, i = c // 26, c % 26
        jax_name = chr(97 + i) + jax_name
    jax_name += getattr(jax_var, "suffix", "")

    if jax_name in util.FORBIDDEN_SDFG_VAR_NAMES:
        jax_name = f"__jace_forbidden_{jax_name}"
    return jax_name


def get_jax_literal_value(lit: jax_core.Atom) -> bool | float | int | np.generic:
    """
    Returns the value a literal is wrapping.

    The function guarantees to return a scalar value.
    """
    if not isinstance(lit, jax_core.Literal):
        raise TypeError(f"Can only extract literals not '{type(lit)}'.")
    val = lit.val
    if isinstance(val, np.ndarray):
        assert val.shape == ()
        return val.max()
    if isinstance(val, (bool, float, int)):
        return val
    raise TypeError(f"Failed to extract value from '{lit}'.")
