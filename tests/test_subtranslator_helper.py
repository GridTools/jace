# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests to check if the sorting algorithm is correct."""

from __future__ import annotations

from jace import translator as jtrans


def test_subtranslatior_managing():
    """Ensures the functionality of the subtranslator managing."""
    from jace.translator.sub_translators import (
        _get_subtranslators_cls,
        add_subtranslator,
    )

    # These are all initial subtranslators
    builtin_subtrans_cls = _get_subtranslators_cls()

    # Definitions of some classes to help.
    class SubTrans1(jtrans.PrimitiveTranslator):
        @classmethod
        def CREATE(cls) -> SubTrans1:
            return SubTrans1()

        @property
        def primitive(self):
            return "non_existing_primitive1"

        def translate_jaxeqn(self) -> None:  # type: ignore[override]  # Arguments
            return None

    class SubTrans2(jtrans.PrimitiveTranslator):
        @classmethod
        def CREATE(cls) -> SubTrans2:
            return SubTrans2()

        @property
        def primitive(self):
            return "non_existing_primitive2"

        def translate_jaxeqn(self) -> None:  # type: ignore[override]  # Arguments
            return None

    # Adding the first subtranslator to the list.
    assert add_subtranslator(SubTrans1)

    curr_subtrans_cls = _get_subtranslators_cls()
    assert len(curr_subtrans_cls) == len(builtin_subtrans_cls) + 1
    assert [SubTrans1, *builtin_subtrans_cls] == curr_subtrans_cls

    # Now adding the second subtranslator
    assert add_subtranslator(SubTrans2)

    curr_subtrans_cls2 = _get_subtranslators_cls()
    assert len(curr_subtrans_cls2) == len(builtin_subtrans_cls) + 2
    assert [SubTrans2, SubTrans1, *builtin_subtrans_cls] == curr_subtrans_cls2
    assert curr_subtrans_cls2 is not curr_subtrans_cls


if __name__ == "__main__":
    test_subtranslatior_managing()
