# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests to check if the sorting algorithm is correct."""

from __future__ import annotations

import re
from typing import Any

import jax
import numpy as np
import pytest

import jace
from jace import translator
from jace.translator import (
    get_regsitered_primitive_translators,
    register_primitive_translator,
)


@pytest.fixture(autouse=True)
def _conserve_builtin_translators():
    """Restores the set of registered subtranslators after a test."""
    initial_translators = translator.get_regsitered_primitive_translators()
    yield
    translator.set_active_primitive_translators_to(initial_translators)


@pytest.fixture()
def no_builtin_translators() -> str:
    """This fixture can be used if the test does not want any builtin translators."""
    initial_translators = translator.get_regsitered_primitive_translators()
    translator.set_active_primitive_translators_to({})
    yield "DUMMY_VALUE"
    translator.set_active_primitive_translators_to(initial_translators)


# These are definitions of some Subtranslators that can be used to test things.
class SubTrans1(translator.PrimitiveTranslator):
    @property
    def primitive(self):
        return "non_existing_primitive1"

    def __call__(self) -> None:  # type: ignore[override]  # Arguments
        raise NotImplementedError


class SubTrans2(translator.PrimitiveTranslator):
    @property
    def primitive(self):
        return "non_existing_primitive2"

    def __call__(self) -> None:  # type: ignore[override]  # Arguments
        raise NotImplementedError


# fmt: off
def SubTrans3_Callable(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError
SubTrans3_Callable.primitive = "non_existing_primitive3" # type: ignore[attr-defined]
# fmt: on


def test_are_subtranslators_imported():
    """Tests if something is inside the list of subtranslators."""
    # Must be adapted if new primitives are implemented.
    assert len(get_regsitered_primitive_translators()) == 37


def test_subtranslatior_managing(no_builtin_translators):
    """Basic functionality of the subtranslators."""
    original_active_subtrans = get_regsitered_primitive_translators()
    assert len(original_active_subtrans) == 0

    # Create the classes.
    sub1 = SubTrans1()
    sub2 = SubTrans2()

    # These are all primitive translators
    prim_translators = [sub1, sub2, SubTrans3_Callable]

    # Add the instances.
    for sub in prim_translators:
        assert register_primitive_translator(sub) is sub

    # Tests if they were correctly registered
    active_subtrans = get_regsitered_primitive_translators()
    for expected in prim_translators:
        assert active_subtrans[expected.primitive] is expected
    assert len(active_subtrans) == 3


def test_subtranslatior_managing_callable(no_builtin_translators):
    """If we add a callable, and have no `.primitive` property defined."""

    def noname_translator_callable(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    # This will not work because `noname_translator_callable()` does not have a `.primitive` attribute.
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"Missing primitive name for '{noname_translator_callable}'"),
    ):
        register_primitive_translator(noname_translator_callable)
    assert len(get_regsitered_primitive_translators()) == 0

    # This works because there is a primitive specified, it will also update the object.
    prim_name = "noname_translator_callable_prim"
    assert register_primitive_translator(noname_translator_callable, primitive=prim_name)
    assert noname_translator_callable.primitive == prim_name


def test_subtranslatior_managing_failing_wrong_name(no_builtin_translators):
    """Tests if how it works with wrong name."""
    sub1 = SubTrans1()
    sub2 = SubTrans2()

    with pytest.raises(
        expected_exception=TypeError,
        match=re.escape(
            f"Translator's primitive '{sub1.primitive}' doesn't match the supplied '{sub2.primitive}'."
        ),
    ):
        register_primitive_translator(sub1, primitive=sub2.primitive)


def test_subtranslatior_managing_overwriting():
    """Tests if we are able to overwrite something."""
    current_add_translator = get_regsitered_primitive_translators()["add"]

    def useless_add_translator(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    useless_add_translator.primitive = "add"

    # This will not work because it is not overwritten.
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            "Explicit override=True needed for primitive 'add' to overwrite existing one."
        ),
    ):
        register_primitive_translator(useless_add_translator)
    assert current_add_translator is get_regsitered_primitive_translators()["add"]

    # Now we use overwrite, thus it will now work.
    assert useless_add_translator is register_primitive_translator(
        useless_add_translator, overwrite=True
    )


def test_subtranslatior_managing_overwriting_2(no_builtin_translators):
    """Again an overwriting test, but this time a bit more complicated."""
    jax.config.update("jax_enable_x64", True)

    trans_cnt = [0]

    @register_primitive_translator(primitive="add")
    def still_but_less_useless_add_translator(*args: Any, **kwargs: Any) -> None:
        trans_cnt[0] += 1
        return

    @jace.jit
    def foo(A):
        B = A + 1
        C = B + 1
        D = C + 1
        return D + 1

    _ = foo.lower(1)
    assert trans_cnt[0] == 4


def test_subtranslatior_managing_decoupling():
    """Shows that we have proper decoupling.

    I.e. changes to the global state, does not affect already annotated functions.
    """
    jax.config.update("jax_enable_x64", True)

    @jace.jit
    def foo(A):
        B = A + 1
        C = B + 1
        D = C + 1
        return D + 1

    @register_primitive_translator(primitive="add", overwrite=True)
    def useless_add_translator(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("The 'useless_add_translator' was called as expected.")

    # Since `foo` was already constructed, a new registering can not change anything.
    A = np.zeros((10, 10))
    assert np.all(foo(A) == 4)

    # But if we now annotate a new function, then we will get the uselss translator
    @jace.jit
    def foo_fail(A):
        B = A + 1
        return B + 1

    with pytest.raises(
        expected_exception=NotImplementedError,
        match=re.escape("The 'useless_add_translator' was called as expected."),
    ):
        _ = foo_fail.lower(A)
