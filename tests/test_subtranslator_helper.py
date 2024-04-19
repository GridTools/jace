"""Implements tests to check if the sorting algorithm is correct.
"""

from typing                     import Collection, Sequence, Union, Type

import jace
from jace   import translator as jtrans


def test_subtranslatior_order_simple():
    """Tests if the ordering of subtranslators works correctly.

    Simple version that only uses priorities.
    """
    from jace.translator.util.subtranslator_helper_order import  sort_subtranslators

    class SimpleSubTrans1(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 0
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def get_priority(self):
            return 1
    # end class(SimpleSubTrans1):

    class SimpleSubTrans2(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 1  # Not last because, default prio is always last.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def get_priority(self):
            return jtrans.JaCeSubTranslatorInterface.DEFAULT_PRIORITY + 1
    # end class(SimpleSubTrans2):

    class SimpleSubTrans3(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 2
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    # end class(SimpleSubTrans3):

    initial_order = [
            SimpleSubTrans3(),
            SimpleSubTrans2(),
            SimpleSubTrans1(),
    ]

    # Now call the function.
    sorted_translators = sort_subtranslators(initial_order)

    # Now we bring the list in expected order.
    expected_order = sorted(initial_order, key=lambda st: st._EXP_ORDER)

    assert all(ist is soll  for ist, soll in zip(sorted_translators, expected_order)), \
            f"Expected order was `{[type(x).__name__  for x in expected_order]}`, but got `{[type(x).__name__  for x in sorted_translators]}`."
    return True
# end def: test_subtranslatior_order_simple


def test_subtranslatior_order_custom1():
    """Tests if the ordering of subtranslators works correctly.

    Interaction of priorities and custom `__lt__()`.
    """
    from jace.translator.util.subtranslator_helper_order import  sort_subtranslators

    class SimpleSubTrans1(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 0
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def get_priority(self):
            return NotImplemented
        def __lt__(self, other):
            return isinstance(other, SimpleSubTrans2)
    # end class(SimpleSubTrans1):

    class SimpleSubTrans2(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 1
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def get_priority(self):
            return NotImplemented
        def __lt__(self, other):
            return True
    # end class(SimpleSubTrans2):

    class SimpleSubTrans3(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 2
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def get_priority(self):
            return NotImplemented
        def __lt__(self, other):
            return False
    # end class(SimpleSubTrans3):

    class SimpleSubTrans4(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 3
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def get_priority(self):
            return jtrans.JaCeSubTranslatorInterface.DEFAULT_PRIORITY + 1
    # end class(SimpleSubTrans4):

    class SimpleSubTrans5(jtrans.JaCeSubTranslatorInterface):
        _EXP_ORDER = 4
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    # end class(SimpleSubTrans5):

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

    assert all(ist is soll  for ist, soll in zip(sorted_translators, expected_order)), \
            f"Expected order was `{[type(x).__name__  for x in expected_order]}`, but got `{[type(x).__name__  for x in sorted_translators]}`."
    return True
# end def: test_subtranslatior_order_custom1



def test_subtranslatior_managing():
    """Esnsures the functionality of the subtranslator managing.
    """
    from jace.translator.sub_translators import  _get_subtranslators_cls, add_subtranslator, rm_subtranslator

    class ValidSubTrans(jtrans.JaCeSubTranslatorInterface):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    #

    class InvalidSubTrans(object):
        def __init__(self):
            ...
        def get_handled_primitives(self) -> Collection[str] | str:
            return "add"
        def can_translate_jaxeqn(self, *args: Any, **kwargs: Any):
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

    # Thest the initial conditions
    init_sub_trans_list = _get_subtranslators_cls(builtins=False)
    init_built_in = _get_subtranslators_cls(with_external=False)
    assert len(init_sub_trans_list) == 0, f"Expected no external subtranslators but found: {init_sub_trans_list}"

    # Now we add the valid subtranslator interface
    assert add_subtranslator(ValidSubTrans), f"Failed to add the `ValidSubTrans`"
    first_sub_trans = _get_subtranslators_cls(builtins=False)





    # Should not include the 
    subTrans = _get_subtranslators_cls(with_external=False)






    assert not add_subtranslator(ValidSubTrans), f"Could add `ValidSubTrans` twice"

















# end def: test_subtranslatior_order_simple






if "__main__" == __name__:
    test_subtranslatior_order_simple()
    test_subtranslatior_order_custom1()
# end(main):



