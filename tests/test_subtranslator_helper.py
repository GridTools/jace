# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements tests to check if the sorting algorithm is correct."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import pytest

from jace import translator as jtrans


def test_subtranslatior_order_simple():
    """Tests if the ordering of subtranslators works correctly.

    Simple version that only uses priorities.
    """
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
    """Tests if the ordering of subtranslators works correctly.

    Interaction of priorities and custom `__lt__()`.
    """
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


def test_subtranslatior_managing():
    """Ensures the functionality of the subtranslator managing."""
    from jace.translator.sub_translators import (
        _get_subtranslators_cls,
        add_subtranslator,
        rm_subtranslator,
    )

    class ValidSubTrans(jtrans.JaCeSubTranslatorInterface):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ValidSubTrans2(jtrans.JaCeSubTranslatorInterface):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class InvalidSubTrans:
        def __init__(self): ...
        def get_handled_primitives(self) -> Collection[str] | str:
            return "add"

        def can_translate_jaxeqn(self, *args: Any, **kwargs: Any):  # noqa: ARG002  # Unused arguments
            return False

        def translate_jaxeqn(self, *args: Any, **kwargs: Any):
            raise NotImplementedError()

        def get_priority(self) -> int:
            return 0

        def has_default_priority(self) -> bool:
            return False

        def __lt__(self, other: Any) -> bool:
            return NotImplemented

        def __eq__(self, other: Any) -> bool:
            return id(self) == id(other)

        def __hash__(self) -> int:
            return id(self)

        def __ne__(self, other: Any) -> bool:
            return NotImplemented

        def __le__(self, other: Any) -> bool:
            return NotImplemented

        def __ge__(self, other: Any) -> bool:
            return NotImplemented

        def __gt__(self, other: Any) -> bool:
            return NotImplemented

    #

    # Test the initial conditions
    builtin_subtrans = _get_subtranslators_cls(with_external=False)
    curr_external_subtrans = _get_subtranslators_cls(builtins=False)
    exp_curr_external_subtrans = []
    assert (
        curr_external_subtrans == exp_curr_external_subtrans
    ), f"Expected no external subtranslators but found: {builtin_subtrans}"
    assert (
        len(builtin_subtrans) != 0
    ), "Expected to have some builtin subtranslator, but there were none."
    assert builtin_subtrans is not _get_subtranslators_cls()  # Ensures no sharing

    # Add a subtranslator to the internal list
    assert add_subtranslator(ValidSubTrans), "Failed to add 'ValidSubTrans'"
    exp_curr_external_subtrans = [ValidSubTrans]
    curr_external_subtrans = _get_subtranslators_cls(builtins=False)
    assert (
        curr_external_subtrans == exp_curr_external_subtrans
    ), f"Wrong subtranslator order, expected '{exp_curr_external_subtrans}' got '{curr_external_subtrans}'."
    assert builtin_subtrans == _get_subtranslators_cls(with_external=False)
    assert _get_subtranslators_cls() == exp_curr_external_subtrans + builtin_subtrans

    # Add a second translator
    assert add_subtranslator(ValidSubTrans2), "Failed to add 'ValidSubTrans2'"
    exp_curr_external_subtrans = [ValidSubTrans2, ValidSubTrans]  # FILO order
    curr_external_subtrans = _get_subtranslators_cls(builtins=False)
    assert (
        exp_curr_external_subtrans == curr_external_subtrans
    ), f"Wrong subtranslator order, expected '{exp_curr_external_subtrans}' got '{curr_external_subtrans}'."
    assert exp_curr_external_subtrans + builtin_subtrans == _get_subtranslators_cls()

    # Now we try to add some translators that will be rejected.
    assert not add_subtranslator(ValidSubTrans)  # Already known
    assert not add_subtranslator(ValidSubTrans2)  # Already known
    assert not add_subtranslator(ValidSubTrans())  # Is an instance
    assert not add_subtranslator(InvalidSubTrans)  # Not implementing interface
    assert exp_curr_external_subtrans + builtin_subtrans == _get_subtranslators_cls()

    # Now we remove a translator from the list.
    assert rm_subtranslator(ValidSubTrans), "Failed to remove 'ValidSubTrans'"
    exp_curr_external_subtrans = [ValidSubTrans2]
    curr_external_subtrans = _get_subtranslators_cls(builtins=False)
    assert (
        curr_external_subtrans == exp_curr_external_subtrans
    ), f"Wrong subtranslator order, expected '{exp_curr_external_subtrans}' got '{curr_external_subtrans}'."
    assert builtin_subtrans == _get_subtranslators_cls(with_external=False)
    assert _get_subtranslators_cls() == exp_curr_external_subtrans + builtin_subtrans

    # Now try to remove it again.
    assert not rm_subtranslator(ValidSubTrans), "Was allowed to remove 'ValidSubTrans' again!"
    with pytest.raises(
        expected_exception=KeyError, match=f"Subtranslator '{type(ValidSubTrans)}' is not known."
    ):
        rm_subtranslator(ValidSubTrans, strict=True)
    #


if __name__ == "__main__":
    test_subtranslatior_order_simple()
    test_subtranslatior_order_custom1()
    test_subtranslatior_managing()
