# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for managing the primitive subtranslators."""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping
from typing import Any

import numpy as np
import pytest

import jace
from jace import translator

from tests import util as testutil


@pytest.fixture(autouse=True)
def _conserve_builtin_translators() -> Generator[None, None, None]:
    """Restores the set of registered subtranslators after a test."""
    initial_translators = translator.get_registered_primitive_translators()
    yield
    testutil.set_active_primitive_translators_to(initial_translators)


@pytest.fixture()
def no_builtin_translators() -> Generator[None, None, None]:  # noqa: PT004 [pytest-missing-fixture-name-underscore] # This is how you should do it: https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#use-fixtures-in-classes-and-modules-with-usefixtures
    """This fixture can be used if the test does not want any builtin translators."""
    initial_translators = testutil.set_active_primitive_translators_to({})
    yield
    testutil.set_active_primitive_translators_to(initial_translators)


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
def primitive_translator_3_callable(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001 [unused-function-argument]
    raise NotImplementedError


@translator.make_primitive_translator("add")
def fake_add_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001 [unused-function-argument]
    raise NotImplementedError("'fake_add_translator()' was called.")


def test_has_pjit():
    print(f"ADDRESS: {translator.get_registered_primitive_translators()['pjit']}")
    print(f"FUN ADDRESS: {translator.primitive_translators.pjit_translator.PJITTranslator}")
    assert "pjit" in translator.get_registered_primitive_translators()


@pytest.mark.usefixtures("no_builtin_translators")
def test_subtranslatior_managing() -> None:
    """Basic functionality of the subtranslators."""
    original_active_subtrans = translator.get_registered_primitive_translators()
    assert len(original_active_subtrans) == 0

    # Create the classes.
    sub1 = SubTrans1()
    sub2 = SubTrans2()

    # These are all primitive translators
    prim_translators = [sub1, sub2, primitive_translator_3_callable]

    # Add the instances.
    for sub in prim_translators:
        assert translator.register_primitive_translator(sub) is sub

    # Tests if they were correctly registered
    active_subtrans = translator.get_registered_primitive_translators()
    for expected in prim_translators:
        assert active_subtrans[expected.primitive] is expected
    assert len(active_subtrans) == 3


def test_subtranslatior_managing_swap() -> None:
    """Tests the `translator.set_active_primitive_translators_to()` functionality."""

    # Allows to compare the structure of dicts.
    def same_structure(d1: Mapping, d2: Mapping) -> bool:
        return d1.keys() == d2.keys() and all(id(d2[k]) == id(d1[k]) for k in d1)

    initial_primitives = translator.get_registered_primitive_translators()
    assert "add" in initial_primitives

    # Generate a set of translators that we swap in
    new_active_primitives = initial_primitives.copy()
    new_active_primitives["add"] = fake_add_translator

    # Now perform the changes.
    old_active = testutil.set_active_primitive_translators_to(new_active_primitives)
    assert (
        new_active_primitives is not translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY
    )
    assert same_structure(old_active, initial_primitives)
    assert same_structure(new_active_primitives, translator.get_registered_primitive_translators())


def test_subtranslatior_managing_callable_annotation() -> None:
    """Test if `translator.make_primitive_translator()` works."""

    prim_name = "non_existing_property"

    @translator.make_primitive_translator(prim_name)
    def non_existing_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001 [unused-function-argument]
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
    def still_useless_but_a_bit_less(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001 [unused-function-argument]
        trans_cnt[0] += 1

    @jace.jit
    def foo(a: int) -> int:
        b = a + 1
        c = b + 1
        d = c + 1
        return d + 1

    with pytest.warns(
        UserWarning,
        match='WARNING: Use of uninitialized transient "e" in state output_processing_stage',
    ):
        _ = foo.lower(1)
    assert trans_cnt[0] == 4


def test_subtranslatior_managing_decoupling() -> None:
    """Shows that we have proper decoupling.

    I.e. changes to the global state, does not affect already annotated functions.
    """

    # This will use the translators that are currently installed.
    @jace.jit
    def foo(a: np.ndarray) -> np.ndarray:
        b = a + np.int32(1)
        c = b + np.int32(1)
        d = c + np.int32(1)
        return d + np.int32(1)

    # Now register the add translator.
    translator.register_primitive_translator(fake_add_translator, overwrite=True)

    # Since `foo` was already constructed, a new registering can not change anything.
    a = np.zeros((10, 10), dtype=np.int32)
    assert np.all(foo(a) == 4)

    # But if we now annotate a new function, then we will get fake translator
    @jace.jit
    def foo_fail(a: np.ndarray) -> np.ndarray:
        b = a + np.int32(1)
        return b + np.int32(1)

    with pytest.raises(
        expected_exception=NotImplementedError,
        match=re.escape("'fake_add_translator()' was called."),
    ):
        _ = foo_fail.lower(a)
