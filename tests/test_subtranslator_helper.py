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


def test_subtranslatior_managing():
    """Ensures the functionality of the subtranslator managing."""
    from jace.translator.sub_translators import (
        _get_subtranslators_cls,
        add_subtranslator,
        rm_subtranslator,
    )

    class ValidSubTrans(jtrans.PrimitiveTranslator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ValidSubTrans2(jtrans.PrimitiveTranslator):
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
    test_subtranslatior_managing()
