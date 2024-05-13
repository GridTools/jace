# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, cast, overload

from jace.util import traits


_T = TypeVar("_T")


@overload
def as_sequence(value: str) -> Iterable[str]: ...


@overload
def as_sequence(value: Iterable[_T]) -> Iterable[_T]: ...


@overload
def as_sequence(value: _T) -> Iterable[_T]: ...


def as_sequence(value: _T | Iterable[_T]) -> Iterable[_T]:
    if traits.is_non_string_iterable(value):
        return value
    return cast(Iterable[_T], [value])


def dataclass_with_default_init(
    _cls: type | None = None,
    *args: Any,
    **kwargs: Any,
) -> type | Callable[[type], type]:
    """The dataclasses `__init__` will now be made available as `__default_init__` if `_cls` define `__init__`.

    Adapted from `https://stackoverflow.com/a/58336722`
    """
    from dataclasses import dataclass

    def wrap(cls: type) -> type:
        # Save the current __init__ and remove it so dataclass will create the default __init__.
        #  But only do something if the class has an `__init__` function.
        has_user_init = hasattr(cls, "__init__")
        if has_user_init:
            user_init = getattr(cls, "__init__", None)
            delattr(cls, "__init__")

        # let dataclass process our class.
        result = dataclass(cls, *args, **kwargs)  # type: ignore[var-annotated]

        # If there is no init function in the original class then, we are done.
        if not has_user_init:
            return result

        # Restore the user's __init__ save the default init to __default_init__.
        result.__default_init__ = result.__init__
        result.__init__ = user_init

        # Just in case that dataclass will return a new instance,
        # (currently, does not happen), restore cls's __init__.
        if result is not cls:
            cls.__init__ = user_init  # type: ignore[misc]

        return result

    # Support both dataclass_with_default_init() and dataclass_with_default_init
    if _cls is None:
        return wrap
    return wrap(_cls)


# Valid name for a jax variable.
VALID_JAX_VAR_NAME: re.Pattern = re.compile("(jax[0-9]+_?)|([a-z]+_?)")

# Valid name for an SDFG variable.
VALID_SDFG_VAR_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

# Valid name for an SDFG itself, includes `SDFGState` objects.
VALID_SDFG_OBJ_NAME: re.Pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")
