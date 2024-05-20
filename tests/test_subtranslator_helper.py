# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests to check if the sorting algorithm is correct."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any

import dace
import jax
import numpy as np
import pytest
from jax import core as jax_core

import jace
from jace import translator
from jace.translator import (
    get_regsitered_primitive_translators,
    register_primitive_translator,
)


@pytest.fixture(autouse=True)
def _conserve_builtin_translators():
    """Decorator that restores the previous state of the build ins."""
    initial_translators = translator.get_regsitered_primitive_translators()
    yield
    translator.set_active_primitive_translators_to(initial_translators)


def _dict_struct(dict_: Mapping[str, Any]) -> Sequence[tuple[str, int]]:
    return tuple(sorted(((k, id(v)) for k, v in dict_.items()), key=lambda X: X[0]))


def test_are_subtranslators_imported():
    """Tests if something is inside the list of subtranslators."""
    assert len(get_regsitered_primitive_translators()) > 1


def test_subtranslatior_managing():
    """Ensures the functionality of the subtranslator managing."""

    # TODO(phimuell): Make this more friendly; See blow
    builtin_subtrans = get_regsitered_primitive_translators()
    builin_struct = _dict_struct(builtin_subtrans)

    class SubTrans1(translator.PrimitiveTranslator):
        @property
        def primitive(self):
            return "non_existing_primitive1"

        def __call__(self) -> None:  # type: ignore[override]  # Arguments
            raise NotImplementedError

    # Ensures that we really return the object unmodified.
    sub_trans1 = register_primitive_translator(SubTrans1())
    assert sub_trans1 is get_regsitered_primitive_translators()["non_existing_primitive1"]

    class SubTrans2(translator.PrimitiveTranslator):
        @property
        def primitive(self):
            return "non_existing_primitive2"

        def __call__(self) -> None:  # type: ignore[override]  # Arguments
            raise NotImplementedError

    # Wrong name
    sub_trans2_instance = SubTrans2()
    with pytest.raises(
        expected_exception=TypeError,
        match=re.escape(
            f"Translator's primitive '{sub_trans2_instance.primitive}' doesn't match the supplied 'not_non_existing_primitive2'."
        ),
    ):
        register_primitive_translator(
            sub_trans2_instance,
            primitive="not_non_existing_primitive2",
        )

    # But if the correct name is specified it works.
    register_primitive_translator(
        sub_trans2_instance,
        primitive="non_existing_primitive2",
    )

    @register_primitive_translator(primitive="non_existing_primitive3")
    def non_existing_primitive_translator_3(
        driver: translator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        raise NotImplementedError

    assert non_existing_primitive_translator_3.primitive == "non_existing_primitive3"

    curr1_subtrans = get_regsitered_primitive_translators()
    curr1_subtrans_mod = get_regsitered_primitive_translators()
    assert curr1_subtrans is not builtin_subtrans
    assert curr1_subtrans is not curr1_subtrans_mod
    assert _dict_struct(curr1_subtrans) != builin_struct
    assert _dict_struct(curr1_subtrans) == _dict_struct(curr1_subtrans_mod)

    for i in [1, 2, 3]:
        pname = f"non_existing_primitive{i}"
        assert pname in curr1_subtrans, f"Expected to find '{pname}'."
        curr1_subtrans_mod.pop(pname)
    assert builin_struct == _dict_struct(curr1_subtrans_mod)

    # Try adding instance and if we can overwrite.
    sub_trans1_instance = SubTrans1()
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            "Explicit override=True needed for primitive 'non_existing_primitive1' to overwrite existing one."
        ),
    ):
        register_primitive_translator(sub_trans1_instance, overwrite=False)

    # Now adding it forcefully, this should also change a lot.
    register_primitive_translator(sub_trans1_instance, overwrite=True)

    curr2_subtrans = get_regsitered_primitive_translators()
    assert curr2_subtrans is not builtin_subtrans
    assert curr2_subtrans is not curr1_subtrans
    assert _dict_struct(curr2_subtrans) != builin_struct
    assert _dict_struct(curr2_subtrans) != _dict_struct(curr1_subtrans)
    assert curr2_subtrans["non_existing_primitive1"] is sub_trans1_instance

    # Try to register a function as translator, that already has a primitive property.
    with pytest.raises(
        expected_exception=TypeError,
        match=re.escape(
            f"Translator's primitive '{non_existing_primitive_translator_3.primitive}' doesn't match the supplied 'non_existing_primitive1'."
        ),
    ):
        register_primitive_translator(
            non_existing_primitive_translator_3,
            primitive="non_existing_primitive1",
            overwrite=False,
        )

    # This would work because it has the same primitive name, but it fails because overwrite is False
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            "Explicit override=True needed for primitive 'non_existing_primitive3' to overwrite existing one."
        ),
    ):
        register_primitive_translator(
            non_existing_primitive_translator_3,
            primitive="non_existing_primitive3",
            overwrite=False,
        )

    register_primitive_translator(
        non_existing_primitive_translator_3, primitive="non_existing_primitive3", overwrite=True
    )


def test_subtranslatior_managing_2():
    """Shows that we are really able to overwrite stuff"""
    jax.config.update("jax_enable_x64", True)

    class NonAddTranslator(translator.PrimitiveTranslator):
        @property
        def primitive(self):
            return "add"

        def __call__(self, *args, **kwargs) -> None:
            raise NotImplementedError("The 'NonAddTranslator' can not translate anything.")

    register_primitive_translator(NonAddTranslator(), overwrite=True)

    @jace.jit
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    with pytest.raises(
        expected_exception=NotImplementedError,
        match=re.escape("The 'NonAddTranslator' can not translate anything."),
    ):
        _ = testee.lower(A, B)


def test_subtranslatior_managing_3():
    """Shows proper decoupling."""
    jax.config.update("jax_enable_x64", True)

    class NonAddTranslator(translator.PrimitiveTranslator):
        @property
        def primitive(self):
            return "add"

        def __call__(self, *args, **kwargs) -> None:
            raise NotImplementedError("The 'NonAddTranslator' can not translate anything at all.")

    used_sub_trans = get_regsitered_primitive_translators()
    used_sub_trans["add"] = NonAddTranslator()

    @jace.jit(sub_translators=used_sub_trans)
    def not_working_test(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    # Now we again remove the add from the list, but this will not have an impact on the `not_working_test()`.
    used_sub_trans.pop("add")

    @jace.jit
    def working_test(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = np.arange(12, dtype=np.float64).reshape((4, 3))
    B = np.full((4, 3), 10, dtype=np.float64)

    with pytest.raises(
        expected_exception=NotImplementedError,
        match=re.escape("The 'NonAddTranslator' can not translate anything at all."),
    ):
        _ = not_working_test.lower(A, B)

    # This works because the
    working_test.lower(A, B)


if __name__ == "__main__":
    test_subtranslatior_managing()
