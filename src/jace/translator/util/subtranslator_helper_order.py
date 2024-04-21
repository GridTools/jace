# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

from jace import translator


def sort_subtranslators(
    subtranslators: Sequence[translator.JaCeSubTranslatorInterface],
) -> Sequence[translator.JaCeSubTranslatorInterface]:
    """Orders the subtranslators according to their priorities.

    The function ensures the following:
    - All subtranslators that have default priority are at the end.
    - All subtranslators whose `get_priority()` returns `NotImplemented`
        are at the begin of the list. These subtranslators are ordered according
        to their `__lt__()` function.
    - All subtranslators whose `get_priority()` function returns an integer are
        in the middle, ordered according to this value.
    """
    if len(subtranslators) <= 1:
        return subtranslators
    subtranslators = sorted(subtranslators, key=_SubtranslatorOrderingHelper)
    assert (len(subtranslators) <= 1) or all(
        subtranslators[i - 1].has_default_priority() <= subtranslators[i].has_default_priority()
        for i in range(1, len(subtranslators))
    )
    return subtranslators


class _SubtranslatorOrderingHelper:
    """Helper class used by `JaxprTranslationDriver` to bring the subtranslators in the correct order.

    Essentially it is a wrapper that contains the additional logic that is
    needed for sorting. This way subclasses does not need to implement it themselves.

    Notes:
        This class does not implement the other comparison function as requested by PEP8.
    """

    def __init__(self, subtranslator: translator.JaCeSubTranslatorInterface):
        assert isinstance(subtranslator, translator.JaCeSubTranslatorInterface)
        self._sub = subtranslator

    def get(self) -> translator.JaCeSubTranslatorInterface:
        return self._sub

    def __lt__(
        self,
        other: _SubtranslatorOrderingHelper,
    ) -> bool:
        # Default priority means that it will always go to the end.
        if self._sub.has_default_priority():
            return False  # `self` has default priority, so it must go to the end.
        if other._sub.has_default_priority():
            return True  # `self` does not have default prio, thus it _must_ go before `other`.
        prio_self = self._sub.get_priority()  # Get the priorities of the subtranslators.
        prio_other = other._sub.get_priority()
        if all(prio is NotImplemented for prio in (prio_self, prio_other)):
            # None has a prio, `self` should decide if it should go first.
            x = self._sub.__lt__(other._sub)
            assert isinstance(x, bool)
            return x
        # In case only one has a priority, we change the order such that the one that implements
        #  a `__lt__()` goes first.
        if prio_self is NotImplemented:
            assert isinstance(prio_other, int)
            return True
        if prio_other is NotImplemented:
            assert isinstance(prio_self, int)
            return False
        assert all(isinstance(prio, int) for prio in (prio_other, prio_self))
        return prio_self < prio_other