# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Module for managing the individual sutranslators.

The high level idea is that there is a "list" of instances of `PrimitiveTranslator`,
which is known as `_CURRENT_SUBTRANSLATORS`.
If not specified the content of this list is used to perform the translation.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Literal, TypeAlias, cast, overload


if TYPE_CHECKING:
    from jace import translator

    # Type alias for distinguish between instances and classes.
    PrimitiveTranslator: TypeAlias = (
        type[translator.PrimitiveTranslator] | translator.PrimitiveTranslator | Callable
    )


# These are all currently used subtranslators that we are used.
_CURRENT_SUBTRANSLATORS: dict[str, translator.PrimitiveTranslator] = {}
_CURRENT_SUBTRANSLATORS_VIEW: types.MappingProxyType[str, translator.PrimitiveTranslator] = (
    types.MappingProxyType(_CURRENT_SUBTRANSLATORS)
)


def add_subtranslators(
    *subtrans: PrimitiveTranslator | None,
    overwrite: bool = False,
) -> None:
    """Adds many subtranslators in one step to Jace's internal list.

    This function is more efficient if many translators should be added in one go.
    Please refer to `add_subtranslator()` for more information.

    Notes:
        If an error during insertion happens the operation is considered a no ops.
    """
    from jace import translator  # Circular import

    global _CURRENT_SUBTRANSLATORS
    global _CURRENT_SUBTRANSLATORS_VIEW

    if len(subtrans) == 0:
        raise ValueError("Not passed any subtranslators.")

    # Why do we do this kind of versioning here or versioning at all?
    #  The cache has to include the set of used subtranslators somehow.
    #  However, as explained in `JaceWrapped.__init__()` the function must make a copy of it.
    #  One way would be to hash the content, i.e. `[(prim_name, id(prim_translator)), ...]`.
    #  But a much simpler idea is to just consider its address, since in 99% of the cases,
    #  the global list is used and not some user supplied list is used we do this versioning.
    #  This allows `JaceWrapped.__init__()` to identify if the current global list of installed
    #  translated is passed to it and it can then prevent the copying.
    #  In the end a code like:
    #       def foo(...): ...
    #       foo1 = jace.jit(foo).lower()   # noqa: ERA001 commented out code
    #       foo2 = jace.jit(foo).lower()   # noqa: ERA001
    #  Should only lower once as it is seen in Jax.
    new_CURRENT_SUBTRANSLATORS = _CURRENT_SUBTRANSLATORS.copy()

    for prim_trans in subtrans:
        # If it is a class instantiate it.
        if inspect.isclass(prim_trans):
            prim_trans = prim_trans()
        prim_trans = cast(translator.PrimitiveTranslator, prim_trans)

        # Test if we know the primitive already
        prim_name: str = prim_trans.primitive
        if (prim_name in _CURRENT_SUBTRANSLATORS) and (not overwrite):
            raise ValueError(f"Tried to add a second translator for primitive '{prim_name}'.")

        # Commit the change to a "staging"
        new_CURRENT_SUBTRANSLATORS[prim_name] = prim_trans

    # Now update the global variables.
    #  Doing it after the loop gives us exception guarantee
    _CURRENT_SUBTRANSLATORS = new_CURRENT_SUBTRANSLATORS
    _CURRENT_SUBTRANSLATORS_VIEW = types.MappingProxyType(_CURRENT_SUBTRANSLATORS)


def add_subtranslator(
    subtrans: PrimitiveTranslator | None = None,
    /,
    overwrite: bool = False,
) -> PrimitiveTranslator | Callable[[PrimitiveTranslator], PrimitiveTranslator]:
    """Adds the subtranslator `subtrans` to Jace's internal list of translators.

    If the primitive is already known an error is generated, however, if `overwrite` is given,
    then `subtrans` will replace the current one.
    In case `subtrans` is a class, the function will instantiate it first.
    Thus, a class must be constructable without arguments.

    Notes:
        Calls to this function will never modify subtranslator lists previously obtained by `get_subtranslators()`!
        Since `subtrans` is returned unmodified, this function can be used to annotate classes.
        For annotating functions use `add_fsubtranslator()`.

    Todo:
        Accept many inputs for bulk update.
        Add functions to clear them or restore the default ones.
    """

    if subtrans is None:
        # It was used as decorator with some argument (currently `overwrite`).
        def wrapper(
            real_subtrans: PrimitiveTranslator,
        ) -> PrimitiveTranslator:
            return add_subtranslator(real_subtrans, overwrite=overwrite)

        return wrapper

    # Forward the call to the bulk insertion.
    #  And always return the original argument.
    add_subtranslators(subtrans, overwrite=overwrite)
    return subtrans


def add_fsubtranslator(
    prim_name: str,
    fun: Callable | None = None,
    /,
    overwrite: bool = False,
) -> PrimitiveTranslator | Callable[[Callable], PrimitiveTranslator]:
    """Convenience function to annotate function and turn them into a translator.

    Adds the `primitive` property to `fun` and register it then as translator.

    Notes:
        Without this function you would had to define the translator function,
            add the `primitive` property to it and then pass it to `add_subtranslator()`.
            This function allows it to do in one step.
    """

    if fun is None:
        # Annotated mode.
        def wrapper(real_fun: Callable) -> PrimitiveTranslator:
            return add_fsubtranslator(prim_name, real_fun, overwrite=overwrite)

        return wrapper

    assert inspect.isfunction(fun)
    if getattr(fun, "primitive", prim_name) != prim_name:
        raise ValueError(f"Passed 'fun' already '{fun.primitive}' as 'primitive' property.")  # type: ignore[attr-defined]

    fun.primitive = prim_name  # type: ignore[attr-defined]
    return add_subtranslator(fun, overwrite=overwrite)


@overload
def get_subtranslators(  # type: ignore[overload-overlap]
    as_mutable: Literal[False] = False,
) -> Mapping[str, translator.PrimitiveTranslator]: ...


@overload
def get_subtranslators(
    as_mutable: Literal[True] = True,
) -> MutableMapping[str, translator.PrimitiveTranslator]: ...


def get_subtranslators(
    as_mutable: bool = False,
) -> (
    Mapping[str, translator.PrimitiveTranslator]
    | MutableMapping[str, translator.PrimitiveTranslator]
):
    """Returns a view of all _currently_ installed primitive translators in Jace.

    By setting `as_mutable` to `True` the function will return a mutable mapping object.
    However, in any case the returned mapping will not be affected by calls that modify
    the internal list of registered primitive translators, i.e. `add_subtranslator()`.

    Notes:
        If `as_mutable` is `False` the function will return an immutable view of the
            registered primitive translator list, thus only a view is created.
            However, if `as_mutable` is `True` a copy is returned.
    """
    if as_mutable:
        # The use case for this is, that a user wants to populate its own list and do some funky stuff.
        #  Without this option, he would first have to make a mutable copy of the map manually,
        #  every fucking time he wants it, so making an option is simpler.
        return _CURRENT_SUBTRANSLATORS.copy()

    # Since we do a versioning in `add_subtranslator()` we do not have to create a new view.
    #  We can just return the global view, this is needed to fix some problems in the caching.
    return _CURRENT_SUBTRANSLATORS_VIEW
