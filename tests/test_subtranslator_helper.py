# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests for managing the primitive subtranslators."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

import jace
from jace import translator
from jace.translator import (
    get_registered_primitive_translators,
    make_primitive_translator,
    register_primitive_translator,
)


@pytest.fixture(autouse=True)
def _conserve_builtin_translators():
    """Restores the set of registered subtranslators after a test."""
    initial_translators = translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY.copy()
    yield
    translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY.clear()
    translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY.update(initial_translators)


@pytest.fixture()
def no_builtin_translators():  # noqa: PT004  # This is how you should do it: https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#use-fixtures-in-classes-and-modules-with-usefixtures
    """This fixture can be used if the test does not want any builtin translators."""
    initial_translators = translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY.copy()
    translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY.clear()
    yield
    translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY.clear()
    translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY.update(initial_translators)


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


@make_primitive_translator("non_existing_callable_primitive3")
def SubTrans3_Callable(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
    raise NotImplementedError


@make_primitive_translator("add")
def fake_add_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
    raise NotImplementedError


def test_are_subtranslators_imported():
    """Tests if something is inside the list of subtranslators."""
    # Must be adapted if new primitives are implemented.
    assert len(get_registered_primitive_translators()) > 0


@pytest.mark.usefixtures("no_builtin_translators")
def test_subtranslatior_managing():
    """Basic functionality of the subtranslators."""
    original_active_subtrans = get_registered_primitive_translators()
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
    active_subtrans = get_registered_primitive_translators()
    for expected in prim_translators:
        assert active_subtrans[expected.primitive] is expected
    assert len(active_subtrans) == 3


def test_subtranslatior_managing_isolation():
    """Tests if `get_registered_primitive_translators()` decouples."""
    assert (
        get_registered_primitive_translators()
        is not translator.primitive_translator._PRIMITIVE_TRANSLATORS_REGISTRY
    )

    initial_primitives = get_registered_primitive_translators()
    assert get_registered_primitive_translators() is not initial_primitives
    assert "add" in initial_primitives, "For this test the 'add' primitive must be registered."
    org_add_prim = initial_primitives["add"]

    initial_primitives["add"] = fake_add_translator
    assert org_add_prim is not fake_add_translator
    assert get_registered_primitive_translators()["add"] is org_add_prim


@pytest.mark.usefixtures("no_builtin_translators")
def test_subtranslatior_managing_callable_annotation():
    """Test if `make_primitive_translator()` works."""

    prim_name = "non_existing_property"

    @make_primitive_translator(prim_name)
    def non_existing_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise NotImplementedError

    assert hasattr(non_existing_translator, "primitive")
    assert non_existing_translator.primitive == prim_name
    assert len(get_registered_primitive_translators()) == 0


def test_subtranslatior_managing_overwriting():
    """Tests if we are able to overwrite something."""
    current_add_translator = get_registered_primitive_translators()["add"]

    @make_primitive_translator("add")
    def useless_add_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise NotImplementedError

    # This will not work because it is not overwritten.
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            "Explicit override=True needed for primitive 'add' to overwrite existing one."
        ),
    ):
        register_primitive_translator(useless_add_translator)
    assert current_add_translator is get_registered_primitive_translators()["add"]

    # Now we use overwrite, thus it will now work.
    assert useless_add_translator is register_primitive_translator(
        useless_add_translator, overwrite=True
    )
    assert useless_add_translator is get_registered_primitive_translators()["add"]


@pytest.mark.usefixtures("no_builtin_translators")
def test_subtranslatior_managing_overwriting_2():
    """Again an overwriting test, but this time a bit more complicated."""

    trans_cnt = [0]

    @register_primitive_translator(overwrite=True)
    @make_primitive_translator("add")
    def still_useless_but_a_bit_less(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        trans_cnt[0] += 1

    @jace.jit
    def foo(A):
        B = A + 1
        C = B + 1
        D = C + 1
        return D + 1

    with pytest.warns(
        UserWarning,
        match=re.escape('Use of uninitialized transient "e" in state output_processing_stage'),
    ):
        _ = foo.lower(1)
    assert trans_cnt[0] == 4


def test_subtranslatior_managing_decoupling():
    """Shows that we have proper decoupling.

    I.e. changes to the global state, does not affect already annotated functions.
    """

    # This will use the translators that are currently installed.
    @jace.jit
    def foo(A):
        B = A + 1
        C = B + 1
        D = C + 1
        return D + 1

    @register_primitive_translator(overwrite=True)
    @make_primitive_translator("add")
    def useless_add_translator(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
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
