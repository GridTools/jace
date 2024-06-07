# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for managing the primitive subtranslators."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import jace
from jace import translator


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


@pytest.fixture(autouse=True)
def _conserve_builtin_translators() -> Generator[None, None, None]:
    """Restores the set of registered subtranslators after a test."""
    initial_translators = translator.get_registered_primitive_translators()
    yield
    translator.set_active_primitive_translators_to(initial_translators)


@pytest.fixture()
def no_builtin_translators() -> Generator[None, None, None]:  # noqa: PT004  # This is how you should do it: https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#use-fixtures-in-classes-and-modules-with-usefixtures
    """This fixture can be used if the test does not want any builtin translators."""
    initial_translators = translator.set_active_primitive_translators_to({})
    yield
    translator.set_active_primitive_translators_to(initial_translators)


# <------------- Definitions needed for the test


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


@translator.make_primitive_translator("non_existing_callable_primitive3")
def SubTrans3_Callable(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
    raise NotImplementedError


@translator.make_primitive_translator("add")
def fake_add_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
    raise NotImplementedError("'fake_add_translator()' was called.")


@pytest.mark.usefixtures("no_builtin_translators")
def test_subtranslatior_managing() -> None:
    """Basic functionality of the subtranslators."""
    original_active_subtrans = translator.get_registered_primitive_translators()
    assert len(original_active_subtrans) == 0

    # Create the classes.
    sub1 = SubTrans1()
    sub2 = SubTrans2()

    # These are all primitive translators
    prim_translators = [sub1, sub2, SubTrans3_Callable]

    # Add the instances.
    for sub in prim_translators:
        assert translator.register_primitive_translator(sub) is sub

    # Tests if they were correctly registered
    active_subtrans = translator.get_registered_primitive_translators()
    for expected in prim_translators:
        assert active_subtrans[expected.primitive] is expected
    assert len(active_subtrans) == 3


def test_subtranslatior_managing_isolation() -> None:
    """Tests if `translator.get_registered_primitive_translators()` protects the internal registry."""
    assert (
        translator.get_registered_primitive_translators()
        is not translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY
    )

    initial_primitives = translator.get_registered_primitive_translators()
    assert translator.get_registered_primitive_translators() is not initial_primitives
    assert "add" in initial_primitives, "For this test the 'add' primitive must be registered."
    org_add_prim = initial_primitives["add"]

    initial_primitives["add"] = fake_add_translator
    assert org_add_prim is not fake_add_translator
    assert translator.get_registered_primitive_translators()["add"] is org_add_prim


def test_subtranslatior_managing_swap() -> None:
    """Tests the `translator.set_active_primitive_translators_to()` functionality."""

    # Allows to compare the structure of dicts.
    def same_structure(
        d1: Mapping,
        d2: Mapping,
    ) -> bool:
        return d1.keys() == d2.keys() and all(id(d2[k]) == id(d1[k]) for k in d1)

    initial_primitives = translator.get_registered_primitive_translators()
    assert "add" in initial_primitives

    # Generate a set of translators that we swap in
    new_active_primitives = initial_primitives.copy()
    new_active_primitives["add"] = fake_add_translator

    # Now perform the changes.
    old_active = translator.set_active_primitive_translators_to(new_active_primitives)
    assert (
        new_active_primitives is not translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY
    )
    assert same_structure(old_active, initial_primitives)
    assert same_structure(new_active_primitives, translator.get_registered_primitive_translators())


def test_subtranslatior_managing_callable_annotation() -> None:
    """Test if `translator.make_primitive_translator()` works."""

    prim_name = "non_existing_property"

    @translator.make_primitive_translator(prim_name)
    def non_existing_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise NotImplementedError

    assert hasattr(non_existing_translator, "primitive")
    assert non_existing_translator.primitive == prim_name


def test_subtranslatior_managing_overwriting() -> None:
    """Tests if we are able to overwrite a translator in the global registry."""
    current_add_translator = translator.get_registered_primitive_translators()["add"]

    # This will not work because overwriting is not activated.
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            "Explicit override=True needed for primitive 'add' to overwrite existing one."
        ),
    ):
        translator.register_primitive_translator(fake_add_translator)
    assert current_add_translator is translator.get_registered_primitive_translators()["add"]

    # Now we use overwrite.
    assert fake_add_translator is translator.register_primitive_translator(
        fake_add_translator, overwrite=True
    )
    assert fake_add_translator is translator.get_registered_primitive_translators()["add"]


@pytest.mark.usefixtures("no_builtin_translators")
def test_subtranslatior_managing_overwriting_2() -> None:
    """Again an overwriting test, but this time a bit more complicated.

    It also shows if the translator was actually called.
    """

    trans_cnt = [0]

    @translator.register_primitive_translator(overwrite=True)
    @translator.make_primitive_translator("add")
    def still_useless_but_a_bit_less(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        trans_cnt[0] += 1
        return

    @jace.jit
    def foo(A: int) -> int:
        B = A + 1
        C = B + 1
        D = C + 1
        return D + 1

    _ = foo.lower(1)
    assert trans_cnt[0] == 4


def test_subtranslatior_managing_decoupling() -> None:
    """Shows that we have proper decoupling.

    I.e. changes to the global state, does not affect already annotated functions.
    """

    # This will use the translators that are currently installed.
    @jace.jit
    def foo(A: int) -> int:
        B = A + 1
        C = B + 1
        D = C + 1
        return D + 1

    # Now register the add translator.
    translator.register_primitive_translator(fake_add_translator, overwrite=True)

    # Since `foo` was already constructed, a new registering can not change anything.
    A = np.zeros((10, 10))
    assert np.all(foo(A) == 4)

    # But if we now annotate a new function, then we will get fake translator
    @jace.jit
    def foo_fail(A):
        B = A + 1
        return B + 1

    with pytest.raises(
        expected_exception=NotImplementedError,
        match=re.escape("'fake_add_translator()' was called."),
    ):
        _ = foo_fail.lower(A)
