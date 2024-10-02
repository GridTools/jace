# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Implements all utility functions that are related to JAX.

Most of the functions defined here allow an unified access to JAX' internal in
a consistent and stable way.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, overload

import dace
import jax
from jax import core as jax_core, dlpack as jax_dlpack

from jace import util


if TYPE_CHECKING:
    import numpy as np


try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None


@dataclasses.dataclass(repr=True, frozen=True, eq=False)
class JaCeVar:
    """
    Replacement for the `jax.Var` class.

    This class can be seen as some kind of substitute `jax.core.Var`. The main
    intention of this class is as an internal representation of values, as they
    are used in JAX, but without the JAX machinery. As abstract values in JAX
    this class has a datatype, which is a `dace.typeclass` instance and a shape.
    In addition it has an optional name, which allows to create variables with
    a certain name using `JaxprTranslationBuilder.add_array()`.

    If it is expected that code must handle both JAX variables and `JaCeVar`
    then the `get_jax_var_*()` functions should be used.

    Args:
        shape: The shape of the variable.
        dtype: The dace datatype of the variable.
        name: Name the variable should have, optional.

    Note:
        If the name of a `JaCeVar` is '_' it is considered a drop variable. The
        definitions of `__hash__` and `__eq__` are in accordance with how JAX
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

    @classmethod
    def from_atom(
        cls,
        jax_var: jax_core.Atom,
        name: str | None,
    ) -> JaCeVar:
        """
        Generates a `JaCeVar` from the JAX variable `jax_var`.

        If `jax_var` is a literal its value is ignored.

        Args:
            jax_var: The variable to process.
            name: The optional name of the variable.
        """
        return cls(
            shape=get_jax_var_shape(jax_var),
            dtype=get_jax_var_dtype(jax_var),
            name=name,
        )


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
            raise TypeError("Can not derive a name from a JAX Literal.")
        case _:
            raise TypeError(
                f"Does not know how to transform '{jax_var}' (type: '{type(jax_var).__name__}') "
                "into a string."
            )


@overload
def get_jax_var_shape(jax_var: jax_core.Atom) -> tuple[int, ...]: ...


@overload
def get_jax_var_shape(jax_var: JaCeVar) -> tuple[int | dace.symbol | str, ...]: ...


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
    # To detect if there is tracing ongoing, we check the internal tracing stack of JAX.
    #  Note that this is highly internal and depends on the precise implementation of
    #  JAX. For that reason we first look at all arguments and check if they are
    #  tracers. Furthermore, it seems that JAX always have a bottom interpreter on the
    #  stack, thus it is empty if `len(...) == 1`!
    #  See also: https://github.com/google/jax/pull/3370
    if any(isinstance(x, jax_core.Tracer) for x in itertools.chain(args, kwargs.values())):
        return True
    trace_stack_height = len(jax._src.core.thread_local_state.trace_state.trace_stack.stack)
    if trace_stack_height == 1:
        return False
    if trace_stack_height > 1:
        return True
    raise RuntimeError("Failed to determine if tracing is ongoing.")


def translate_dtype(dtype: Any) -> dace.typeclass:
    """Turns a JAX datatype into a DaCe datatype."""
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
        jax_name_map: A mapping of all JAX variables that were already named.

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

    # This code is taken from JAX so it will generate similar ways, the difference is
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
    # In previous versions of JAX literals were always 0-dim arrays, but it seems
    #  that in newer versions the values are either arrays or scalars.
    #  I saw both thus we have to keep both branches.
    if util.is_array(val):
        assert val.shape == ()
        return val.dtype.type(val.max())
    if util.is_scalar(val):
        return val
    raise TypeError(f"Failed to extract value from '{lit}' ('{val}' type: {type(val).__name__}).")


def parse_backend_jit_option(
    backend: str | dace.DeviceType,
) -> dace.DeviceType:
    """Turn JAX' `backend` option into the proper DaCe device type."""
    if isinstance(backend, dace.DeviceType):
        return backend
    match backend:
        case "cpu" | "CPU":
            return dace.DeviceType.CPU
        case "gpu" | "GPU":
            return dace.DeviceType.GPU
        case "fpga" | "FPGA":
            return dace.DeviceType.FPGA
        case "tpu" | "TPU":
            raise NotImplementedError("TPU are not supported.")
        case _:
            raise ValueError(f"Could not parse the backend '{backend}'.")


def move_into_jax_array(
    arr: Any,
    copy: bool | None = False,
) -> jax.Array:
    """
    Moves `arr` into a JAX array using DLPack format.

    By default `copy` is set to `False`, it is the responsibility of the caller
    to ensure that the underlying buffer is not modified later.

    Args:
        arr: The array to move into a JAX array.
        copy: Should a copy be made; defaults to `False`.
    """
    if isinstance(arr, jax.Array):
        return arr
    # In newer version it is no longer needed to pass a capsule.
    return jax_dlpack.from_dlpack(arr, copy=copy)  # type: ignore[attr-defined]  # `from_dlpack` is not found.
