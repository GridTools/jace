# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests to check if the sorting algorithm is correct."""

from __future__ import annotations

from jace import translator as jtrans


def test_subtranslatior_order_simple():
    """This test is to ensure that `sortSubtranslators()` works correctly."""
    from jace.translator.util.subtranslator_helper_order import sort_subtranslators

    class SimpleSubTrans1(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_priority(self):
            return 1

    class SimpleSubTrans2(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 1  # Not last because, default prio is always last.

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_priority(self):
            return jtrans.JaCeSubTranslatorInterface.DEFAULT_PRIORITY + 1

    class SimpleSubTrans3(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 2

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    initial_order = [
        SimpleSubTrans3(),
        SimpleSubTrans2(),
        SimpleSubTrans1(),
    ]

    # Now call the function.
    sorted_translators = sort_subtranslators(initial_order)

    # Now we bring the list in expected order.
    expected_order = sorted(initial_order, key=lambda st: st._EXP_ORDER)

    assert all(
        got_ord is exp_ord
        for got_ord, exp_ord in zip(sorted_translators, expected_order, strict=False)
    ), f"Expected order was `{[type(x).__name__  for x in expected_order]}`, but got `{[type(x).__name__  for x in sorted_translators]}`."
    return True


def test_subtranslatior_order_custom1():
    from jace.translator.util.subtranslator_helper_order import sort_subtranslators

    class SimpleSubTrans1(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_priority(self):
            return NotImplemented

        def __lt__(self, other):
            return isinstance(other, SimpleSubTrans2)

    class SimpleSubTrans2(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_priority(self):
            return NotImplemented

        def __lt__(self, other):
            return True

    class SimpleSubTrans3(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 2

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_priority(self):
            return NotImplemented

        def __lt__(self, other):
            return False

    class SimpleSubTrans4(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 3

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_priority(self):
            return jtrans.JaCeSubTranslatorInterface.DEFAULT_PRIORITY + 1

    class SimpleSubTrans5(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 4

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    assert SimpleSubTrans2() < SimpleSubTrans1()

    initial_order = [
        SimpleSubTrans5(),
        SimpleSubTrans4(),
        SimpleSubTrans3(),
        SimpleSubTrans2(),
        SimpleSubTrans1(),
    ]

    # Now call the function.
    sorted_translators = sort_subtranslators(initial_order)

    # Now we bring the list in expected order.
    expected_order = sorted(initial_order, key=lambda st: st._EXP_ORDER)

    assert all(
        got_ord is exp_ord
        for got_ord, exp_ord in zip(sorted_translators, expected_order, strict=False)
    ), f"Expected order was `{[type(x).__name__  for x in expected_order]}`, but got `{[type(x).__name__  for x in sorted_translators]}`."
    return True


if __name__ == "__main__":
    test_subtranslatior_order_simple()
    test_subtranslatior_order_custom1()
