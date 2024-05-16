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
from inspect import isclass, isfunction
from typing import Any

import dace
import pytest
from jax import core as jax_core

from jace import translator as jtrans
from jace.translator import (
    add_fsubtranslator,
    add_subtranslator,
    get_subtranslators,
)


@pytest.fixture(autouse=True)
def _conserve_builtin_translators():
    """Decorator that preserves the initial list of built in translators.

    Todo:
        Come up with something better/nicer.
    """
    initial_translators = get_subtranslators()
    yield
    jtrans.add_subtranslators(*initial_translators.values(), overwrite=True)


def _dict_struct(dict_: Mapping[str, Any]) -> Sequence[tuple[str, int]]:
    return tuple(sorted(((k, id(v)) for k, v in dict_.items()), key=lambda X: X[0]))


def test_are_subtranslators_imported():
    """Tests if something is inside the list of subtranslators."""
    assert len(get_subtranslators()) > 1


def test_subtranslatior_managing():
    """Ensures the functionality of the subtranslator managing."""

    # TODO(phimuell): Make this more friendly; See blow
    builtin_subtrans = get_subtranslators()
    builin_struct = _dict_struct(builtin_subtrans)

    class SubTrans1(jtrans.PrimitiveTranslator):
        @property
        def primitive(self):
            return "non_existing_primitive1"

        def __call__(self) -> None:  # type: ignore[override]  # Arguments
            raise NotImplementedError

    # Ensures that we really return the object unmodified.
    SubTrans1_ = add_subtranslator(SubTrans1)
    assert isclass(SubTrans1_)
    assert SubTrans1_ is SubTrans1

    @add_subtranslator(overwrite=True)
    class SubTrans2(jtrans.PrimitiveTranslator):
        @property
        def primitive(self):
            return "non_existing_primitive2"

        def __call__(self) -> None:  # type: ignore[override]  # Arguments
            raise NotImplementedError

    assert isclass(SubTrans2)

    @add_fsubtranslator("non_existing_primitive3")
    def non_existing_primitive_translator_3(
        driver: jtrans.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> dace.SDFGState | None:
        raise NotImplementedError

    assert isfunction(non_existing_primitive_translator_3)
    assert non_existing_primitive_translator_3.primitive == "non_existing_primitive3"

    curr1_subtrans = get_subtranslators()
    curr1_subtrans_mod = get_subtranslators(as_mutable=True)
    assert curr1_subtrans is not builtin_subtrans
    assert curr1_subtrans is not curr1_subtrans_mod
    assert _dict_struct(curr1_subtrans) != builin_struct
    assert _dict_struct(curr1_subtrans) == _dict_struct(curr1_subtrans_mod)

    for i in [1, 2, 3]:
        pname = f"non_existing_primitive{i}"
        assert pname in curr1_subtrans, f"Expected to find '{pname}'."
        curr1_subtrans_mod.pop(pname)
    assert builin_struct == _dict_struct(curr1_subtrans_mod)
    assert curr1_subtrans is get_subtranslators()

    # Try adding instance and if we can overwrite.
    sub_trans1_instance = SubTrans1()
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            "Tried to add a second translator for primitive 'non_existing_primitive1'."
        ),
    ):
        add_subtranslator(sub_trans1_instance, overwrite=False)

    # Now adding it forcefully, this should also change a lot.
    add_subtranslator(sub_trans1_instance, overwrite=True)

    curr2_subtrans = get_subtranslators()
    assert curr2_subtrans is not builtin_subtrans
    assert curr2_subtrans is not curr1_subtrans
    assert _dict_struct(curr2_subtrans) != builin_struct
    assert _dict_struct(curr2_subtrans) != _dict_struct(curr1_subtrans)
    assert curr2_subtrans["non_existing_primitive1"] is sub_trans1_instance

    # Try to answer a function as translator, that already has a primitive property.
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape("Passed 'fun' already 'non_existing_primitive3' as 'primitive' property."),
    ):
        add_fsubtranslator(
            "non_existing_primitive1", non_existing_primitive_translator_3, overwrite=False
        )

    # This would work because it has the same primitive name, but it fails because overwrite is False
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            "Tried to add a second translator for primitive 'non_existing_primitive3'."
        ),
    ):
        add_fsubtranslator(
            "non_existing_primitive3", non_existing_primitive_translator_3, overwrite=False
        )

    add_fsubtranslator(
        "non_existing_primitive3", non_existing_primitive_translator_3, overwrite=True
    )


if __name__ == "__main__":
    test_subtranslatior_managing()
