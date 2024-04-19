"""Implements tests to check if the sorting algorithm is correct.
"""

from typing                     import Collection, Sequence, Union

import jace
from jace   import translator as jtrans


def test_subtranslatior_order_simple():
    """This test is to ensure that `sortSubtranslators()` works correctly.
    """
    from jace.translator.util.subtranslator_helper_order import  sort_subtranslators

    class SimpleSubTrans1(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 0
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def getPriority(self):
            return 1
    # end class(SimpleSubTrans1):

    class SimpleSubTrans2(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 1  # Not last because, default prio is always last.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def getPriority(self):
            return jtrans.JaCeSubTranslatorInterface.DEFAULT_PRIORITY + 1
    # end class(SimpleSubTrans2):

    class SimpleSubTrans3(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 2
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    # end class(SimpleSubTrans3):

    initialOrder = [
            SimpleSubTrans3(),
            SimpleSubTrans2(),
            SimpleSubTrans1(),
    ]

    # Now call the function.
    sortedTranslators = sortSubtranslators(initialOrder)

    # Now we bring the list in expected order.
    expectedOrder = sorted(initialOrder, key=lambda st: st._EXP_ORDER)

    assert all(ist is soll  for ist, soll in zip(sortedTranslators, expectedOrder)), \
            f"Expected order was `{[type(x).__name__  for x in expectedOrder]}`, but got `{[type(x).__name__  for x in sortedTranslators]}`."
    return True
# end def: test_subtranslatior_order_simple


def test_subtranslatior_order_custom1():
    from Jax2DaCe.translator.util._subtranslator_helper_order       import sortSubtranslators

    class SimpleSubTrans1(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 0
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def getPriority(self):
            return NotImplemented
        def __lt__(self, other):
            return isinstance(other, SimpleSubTrans2)
    # end class(SimpleSubTrans1):

    class SimpleSubTrans2(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 1
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def getPriority(self):
            return NotImplemented
        def __lt__(self, other):
            return True
    # end class(SimpleSubTrans2):

    class SimpleSubTrans3(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 2
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def getPriority(self):
            return NotImplemented
        def __lt__(self, other):
            return False
    # end class(SimpleSubTrans3):

    class SimpleSubTrans4(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 3
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def getPriority(self):
            return jtrans.JaCeSubTranslatorInterface.DEFAULT_PRIORITY + 1
    # end class(SimpleSubTrans4):

    class SimpleSubTrans5(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 4
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    # end class(SimpleSubTrans5):

    assert SimpleSubTrans2() < SimpleSubTrans1()

    initialOrder = [
            SimpleSubTrans5(),
            SimpleSubTrans4(),
            SimpleSubTrans3(),
            SimpleSubTrans2(),
            SimpleSubTrans1(),
    ]

    # Now call the function.
    sortedTranslators = sortSubtranslators(initialOrder)

    # Now we bring the list in expected order.
    expectedOrder = sorted(initialOrder, key=lambda st: st._EXP_ORDER)

    assert all(ist is soll  for ist, soll in zip(sortedTranslators, expectedOrder)), \
            f"Expected order was `{[type(x).__name__  for x in expectedOrder]}`, but got `{[type(x).__name__  for x in sortedTranslators]}`."
    return True
# end def: test_subtranslatior_order_custom1


if "__main__" == __name__:
    test_subtranslatior_order_simple()
    test_subtranslatior_order_custom1()
# end(main):



