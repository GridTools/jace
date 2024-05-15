# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests to check if the sorting algorithm is correct."""

from __future__ import annotations

import re

import pytest

from jace import translator as jtrans


def test_subtranslatior_managing():
    """Ensures the functionality of the subtranslator managing."""
    from jace.translator import (
        add_subtranslator,
        get_subtranslators_cls,
    )

    # These are all initial subtranslators
    builtin_subtrans_cls = get_subtranslators_cls()

    # Definitions of some classes to help.
    class SubTrans1(jtrans.PrimitiveTranslator):
        @classmethod
        def build_translator(cls) -> SubTrans1:
            return SubTrans1()

        @property
        def primitive(self):
            return "non_existing_primitive1"

        def translate_jaxeqn(self) -> None:  # type: ignore[override]  # Arguments
            return None

    class SubTrans2(jtrans.PrimitiveTranslator):
        @classmethod
        def build_translator(cls) -> SubTrans2:
            return SubTrans2()

        @property
        def primitive(self):
            return "non_existing_primitive2"

        def translate_jaxeqn(self) -> None:  # type: ignore[override]  # Arguments
            return None

    assert SubTrans1 != SubTrans2

    # Adding the first subtranslator to the list.
    add_subtranslator(SubTrans1)

    curr_subtrans_cls = get_subtranslators_cls()
    assert len(curr_subtrans_cls) == len(builtin_subtrans_cls) + 1
    assert all(
        type(exp) == type(got)
        for exp, got in zip([SubTrans1, *builtin_subtrans_cls], curr_subtrans_cls)
    )

    # Now adding the second subtranslator
    add_subtranslator(SubTrans2)

    curr_subtrans_cls2 = get_subtranslators_cls()
    assert len(curr_subtrans_cls2) == len(builtin_subtrans_cls) + 2
    assert [SubTrans2, SubTrans1, *builtin_subtrans_cls] == curr_subtrans_cls2
    assert curr_subtrans_cls2 is not curr_subtrans_cls

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            f"Tried to add '{type(SubTrans1).__name__}' twice to the list of known primitive translators."
        ),
    ):
        add_subtranslator(SubTrans2)

    @add_subtranslator
    class SubTrans3(jtrans.PrimitiveTranslator):
        @classmethod
        def build_translator(cls) -> SubTrans2:
            return SubTrans2()

        @property
        def primitive(self):
            return "non_existing_primitive2"

        def translate_jaxeqn(self) -> None:  # type: ignore[override]  # Arguments
            return None

    curr_subtrans_cls3 = get_subtranslators_cls()
    assert len(curr_subtrans_cls3) == len(builtin_subtrans_cls) + 3
    assert [SubTrans3, SubTrans2, SubTrans1, *builtin_subtrans_cls] == curr_subtrans_cls3

    # Adding version 1 again, but this time using overwrite
    add_subtranslator(SubTrans1, overwrite=True)
    curr_subtrans_cls4 = get_subtranslators_cls()
    assert len(curr_subtrans_cls3) == len(curr_subtrans_cls4)
    assert [SubTrans1, SubTrans3, SubTrans2, *builtin_subtrans_cls] == curr_subtrans_cls4


if __name__ == "__main__":
    test_subtranslatior_managing()
