# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements some tests of the subtranslator driver."""

from __future__ import annotations

import re

import dace
import numpy as np
import pytest
from dace.data import Array

from jace import translator, util
from jace.util import JaCeVar


# These are some Jace variables that we use inside the tests
#  Unnamed arrays
array1 = JaCeVar((10, 12), dace.float64)
array2 = JaCeVar((10, 13), dace.float32)
array3 = JaCeVar((11, 16), dace.int64)

#  Unnamed scalars
scal1 = JaCeVar((), dace.float16)
scal2 = JaCeVar((), dace.float32)
scal3 = JaCeVar((), dace.int64)

# Named variables
narray = JaCeVar((10,), dace.float16, "narr")
nscal = JaCeVar((), dace.int32, "nscal")


@pytest.fixture()
def translation_driver():
    """Returns an allocated driver instance."""
    name = "fixture_driver"
    driver = translator.JaxprTranslationDriver(
        sub_translators=translator.get_regsitered_primitive_translators()
    )
    driver._allocate_translation_ctx(name=name)
    return driver


def test_driver_alloc() -> None:
    """Tests the state right after allocation.

    Does not use the fixture because it does it on its own.
    """
    driver = translator.JaxprTranslationDriver(
        sub_translators=translator.get_regsitered_primitive_translators()
    )
    assert not driver.is_allocated(), "Driver was created allocated."
    assert len(driver._ctx_stack) == 0

    # The reserved names will be tested in `test_driver_fork()`.
    sdfg_name = "qwertzuiopasdfghjkl"
    driver._allocate_translation_ctx(name=sdfg_name)
    assert len(driver._ctx_stack) == 1
    assert driver.is_root_translator()

    sdfg: dace.SDFG = driver.sdfg

    assert driver._ctx.sdfg is sdfg
    assert driver.sdfg.name == sdfg_name
    assert sdfg.number_of_nodes() == 1
    assert sdfg.number_of_edges() == 0
    assert sdfg.start_block is driver._ctx.start_state
    assert driver._terminal_sdfg_state is driver._ctx.start_state


def test_driver_variable_alloc_auto_naming(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests simple variable allocation."""
    for i, var in enumerate([array1, array2, scal1, array3, scal2, scal3]):
        sdfg_name = translation_driver.add_array(var, update_var_mapping=True)
        sdfg_var = translation_driver.get_array(sdfg_name)
        assert sdfg_name == chr(97 + i)
        assert isinstance(sdfg_var, Array)  # Everything is now an array
        assert sdfg_var.shape == ((1,) if var.shape == () else var.shape)
        assert sdfg_var.dtype == var.dtype


def test_driver_variable_alloc_mixed_naming(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests the naming in a mixed setting.

    If `update_var_mapping=True` is given, then the naming will skip variables, see also `test_driver_variable_alloc_mixed_naming2()`.
    """
    #                        *       b       c       d      *      f      g
    for i, var in enumerate([narray, array1, array2, scal1, nscal, scal2, scal3]):
        sdfg_name = translation_driver.add_array(var, update_var_mapping=True)
        sdfg_var = translation_driver.get_array(sdfg_name)
        if var.name is None:
            assert sdfg_name == chr(97 + i)
        else:
            assert sdfg_name == var.name
        assert isinstance(sdfg_var, Array)  # Everything is now an array
        assert sdfg_var.shape == ((1,) if var.shape == () else var.shape)
        assert sdfg_var.dtype == var.dtype


def test_driver_variable_alloc_mixed_naming2(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests the naming in a mixed setting.

    This time we do not use `update_var_mapping=True`, instead it now depends on the name.
    This means that automatic naming will now again include all, letters, but not in a linear order.
    """
    letoff = 0
    #             *      a        b       c      *      d     e
    for var in [narray, array1, array2, scal1, nscal, scal2, scal3]:
        sdfg_name = translation_driver.add_array(var, update_var_mapping=var.name is None)
        sdfg_var = translation_driver.get_array(sdfg_name)
        if var.name is None:
            assert sdfg_name == chr(97 + letoff)
            letoff += 1
        else:
            assert sdfg_name == var.name
        assert isinstance(sdfg_var, Array)  # Everything is now an array
        assert sdfg_var.shape == ((1,) if var.shape == () else var.shape)
        assert sdfg_var.dtype == var.dtype


def test_driver_variable_alloc_prefix_naming(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Using the prefix to name variables."""
    prefix_1 = "__my_special_prefix"
    exp_name_1 = prefix_1 + "a"
    sdfg_name_1 = translation_driver.add_array(
        array1, name_prefix=prefix_1, update_var_mapping=False
    )
    assert exp_name_1 == sdfg_name_1

    # Because `update_var_mapping` is `False` above, 'a' will be reused.
    prefix_2 = "__my_special_prefix_second_"
    exp_name_2 = prefix_2 + "a"
    sdfg_name_2 = translation_driver.add_array(
        array1, name_prefix=prefix_2, update_var_mapping=False
    )
    assert exp_name_2 == sdfg_name_2

    # Now we use a named variables, which are also affected.
    prefix_3 = "__my_special_prefix_third_named_"
    exp_name_3 = prefix_3 + nscal.name  # type: ignore[operator]  # `.name` is not `None`.
    sdfg_name_3 = translation_driver.add_array(
        nscal, name_prefix=prefix_3, update_var_mapping=False
    )
    assert exp_name_3 == sdfg_name_3


def test_driver_variable_alloc_auto_naming_wrapped(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests the variable naming if we have more than 26 variables."""
    single_letters = [chr(x) for x in range(97, 123)]
    i = 0
    for let1 in ["", *single_letters[1:]]:  # Note `z` is followed by `ba` and not by `aa`.
        for let2 in single_letters:
            i += 1
            # Create a variable and enter it into the variable naming.
            var = JaCeVar(shape=(19, 19), dtype=dace.float64)
            sdfg_name = translation_driver.add_array(arg=var, update_var_mapping=True)
            mapped_name = translation_driver.map_jax_var_to_sdfg(var)
            assert (
                sdfg_name == mapped_name
            ), f"Mapping for '{var}' failed, expected '{sdfg_name}' got '{mapped_name}'."

            # Get the name that we really expect, we must also handle some situations.
            exp_name = let1 + let2
            if exp_name in util.FORBIDDEN_SDFG_VAR_NAMES:
                exp_name = "__jace_forbidden_" + exp_name
            assert (
                exp_name == sdfg_name
            ), f"Automated naming failed, expected '{exp_name}' but got '{sdfg_name}'."


def test_driver_nested(translation_driver: translator.JaxprTranslationDriver) -> None:
    """Tests the ability of the nesting of the driver."""

    # Now add a variable to the current subtext.
    name_1 = translation_driver.add_array(array1, update_var_mapping=True)
    assert name_1 == "a"
    assert translation_driver.map_jax_var_to_sdfg(array1) == name_1

    # For the sake of doing it add a new state to the SDFG.
    translation_driver.append_new_state("sake_state")
    assert translation_driver.sdfg.number_of_nodes() == 2
    assert translation_driver.sdfg.number_of_edges() == 1

    # Now we go one subcontext deeper; note we do this manually which should not be done.
    translation_driver._allocate_translation_ctx("driver")
    assert len(translation_driver._ctx_stack) == 2
    assert translation_driver.sdfg.name == "driver"
    assert translation_driver.sdfg.number_of_nodes() == 1
    assert translation_driver.sdfg.number_of_edges() == 0
    assert not translation_driver.is_root_translator()

    # Because we have a new SDFG the mapping to previous SDFG does not work,
    #  regardless the fact that it still exists.
    with pytest.raises(
        expected_exception=KeyError,
        match=re.escape(
            f"Jax variable '{array1}' was supposed to map to '{name_1}', but no such SDFG variable is known."
        ),
    ):
        _ = translation_driver.map_jax_var_to_sdfg(array1)

    # Because the SDFGs are distinct it is possible to add `array1` to the nested one.
    #  However, it is not able to update the mapping.
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            f"Tried to create the mapping '{array1} -> {name_1}', but the variable is already mapped."
        ),
    ):
        _ = translation_driver.add_array(array1, update_var_mapping=True)
    assert name_1 not in translation_driver.sdfg.arrays

    # Without updating the mapping it is possible create the variable.
    assert name_1 == translation_driver.add_array(array1, update_var_mapping=False)

    # Now add a new variable, the map is shared, so a new name will be generated.
    name_2 = translation_driver.add_array(array2, update_var_mapping=True)
    assert name_2 == "b"
    assert name_2 == translation_driver.map_jax_var_to_sdfg(array2)

    # Now we go one stack level back.
    translation_driver._clear_translation_ctx()
    assert len(translation_driver._ctx_stack) == 1
    assert translation_driver.sdfg.number_of_nodes() == 2
    assert translation_driver.sdfg.number_of_edges() == 1

    # Again the variable that was declared in the last stack is now no longer present.
    #  Note if the nested SDFG was integrated into the parent SDFG it would be accessible
    with pytest.raises(
        expected_exception=KeyError,
        match=re.escape(
            f"Jax variable '{array2}' was supposed to map to '{name_2}', but no such SDFG variable is known."
        ),
    ):
        _ = translation_driver.map_jax_var_to_sdfg(array2)
    assert name_2 == translation_driver._jax_name_map[array2]

    # Now add a new variable, since the map is shared, we will now get the next name.
    name_3 = translation_driver.add_array(array3, update_var_mapping=True)
    assert name_3 == "c"
    assert name_3 == translation_driver.map_jax_var_to_sdfg(array3)


def test_driver_append_state(translation_driver: translator.JaxprTranslationDriver) -> None:
    """Tests the functionality of appending states."""
    sdfg: dace.SDFG = translation_driver.sdfg

    terminal_state_1: dace.SDFGState = translation_driver.append_new_state("terminal_state_1")
    assert sdfg.number_of_nodes() == 2
    assert sdfg.number_of_edges() == 1
    assert terminal_state_1 is translation_driver._terminal_sdfg_state
    assert translation_driver._terminal_sdfg_state is translation_driver._ctx.terminal_state
    assert translation_driver._ctx.start_state is sdfg.start_block
    assert translation_driver._ctx.start_state is not terminal_state_1
    assert next(iter(sdfg.edges())).src is sdfg.start_block
    assert next(iter(sdfg.edges())).dst is terminal_state_1

    # Specifying an explicit append state that is the terminal should also update the terminal state of the driver.
    terminal_state_2: dace.SDFGState = translation_driver.append_new_state(
        "terminal_state_2", prev_state=terminal_state_1
    )
    assert sdfg.number_of_nodes() == 3
    assert sdfg.number_of_edges() == 2
    assert terminal_state_2 is translation_driver._terminal_sdfg_state
    assert sdfg.out_degree(terminal_state_1) == 1
    assert sdfg.out_degree(terminal_state_2) == 0
    assert sdfg.in_degree(terminal_state_2) == 1
    assert next(iter(sdfg.in_edges(terminal_state_2))).src is terminal_state_1

    # Specifying a previous node that is not the terminal state should not do anything.
    non_terminal_state: dace.SDFGState = translation_driver.append_new_state(
        "non_terminal_state", prev_state=terminal_state_1
    )
    assert translation_driver._terminal_sdfg_state is not non_terminal_state
    assert sdfg.in_degree(non_terminal_state) == 1
    assert sdfg.out_degree(non_terminal_state) == 0
    assert next(iter(sdfg.in_edges(non_terminal_state))).src is terminal_state_1


def test_driver_variable_multiple_variables(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """A simple test in which we try to add a variable that are known, but with a different name."""
    # Now we will add `array1` and then different ways of updating it.
    narray1: str = translation_driver.add_array(array1, update_var_mapping=True)

    # It will fail if we use the prefix, because we also want to update.
    prefix = "__jace_prefix"
    prefix_expected_name = prefix + narray1
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            f"Tried to create the mapping '{array1} -> {prefix_expected_name}', but the variable is already mapped."
        ),
    ):
        _ = translation_driver.add_array(array1, update_var_mapping=True, name_prefix=prefix)
    assert prefix_expected_name not in translation_driver.sdfg.arrays

    # But if we do not want to update it then it works.
    prefix_sdfg_name = translation_driver.add_array(
        array1, update_var_mapping=False, name_prefix=prefix
    )
    assert prefix_expected_name in translation_driver.sdfg.arrays
    assert narray1 == translation_driver.map_jax_var_to_sdfg(array1)


def test_driver_variable_invalid_prefix(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Use invalid prefix."""
    # It will fail if we use the prefix, because we also want to update.
    for iprefix in ["0_", "_ja ", "_!"]:
        with pytest.raises(
            expected_exception=ValueError,
            match=re.escape(f"add_array({array1}): Supplied invalid prefix '{iprefix}'."),
        ):
            _ = translation_driver.add_array(array1, update_var_mapping=False, name_prefix=iprefix)
        assert len(translation_driver.sdfg.arrays) == 0


def test_driver_variable_alloc_list(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests part of the `JaxprTranslationDriver::create_jax_var_list()` api."""
    var_list_1 = [array1, nscal, scal2]
    exp_names_1 = ["a", nscal.name, "c"]

    res_names_1 = translation_driver.create_jax_var_list(
        var_list_1,
        update_var_mapping=True,
    )
    assert len(translation_driver.arrays) == 3
    assert res_names_1 == exp_names_1

    # Now a mixture of the collection and creation.
    var_list_2 = [array2, nscal, scal1]
    exp_names_2 = ["d", nscal.name, "e"]

    res_names_2 = translation_driver.create_jax_var_list(
        var_list_2,
        update_var_mapping=True,
    )
    assert res_names_2 == exp_names_2
    assert len(translation_driver.arrays) == 5


@pytest.mark.skip(reason="'create_jax_var_list()' does not clean up in case of an error.")
def test_driver_variable_alloc_list_cleaning(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests part of the `JaxprTranslationDriver::create_jax_var_list()` api.

    It will fail because `update_var_mapping=False` thus the third variable will
    cause an error because it is proposed to `a`, which is already used.
    """
    var_list = [array1, nscal, scal2]
    exp_names = ["a", nscal.name, "c"]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"add_array({scal2}): The proposed name 'a', is used."),
    ):
        res_names = translation_driver.create_jax_var_list(var_list)

    # This currently fails, because the `create_jax_var_list()` function does not clean up.
    assert len(translation_driver.arrays) == 0


def test_driver_variable_alloc_list_prevent_creation(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests part of the `JaxprTranslationDriver::create_jax_var_list()` api.

    It will test the `prevent_creation` flag.
    """
    # First create a variable.
    translation_driver.add_array(array1, update_var_mapping=True)
    assert len(translation_driver.arrays) == 1

    # Now create the variables
    var_list = [array1, array2]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"'prevent_creation' given but have to create '{array2}'."),
    ):
        translation_driver.create_jax_var_list(
            var_list,
            prevent_creation=True,
        )
    assert len(translation_driver.arrays) == 1
    assert translation_driver.map_jax_var_to_sdfg(array1) == "a"


@pytest.mark.skip(reason="'create_jax_var_list()' does not clean up in case of an error.")
def test_driver_variable_alloc_list_only_creation(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests part of the `JaxprTranslationDriver::create_jax_var_list()` api.

    It will test the `only_creation` flag.
    """
    # First create a variable.
    translation_driver.add_array(array1, update_var_mapping=True)
    assert len(translation_driver.arrays) == 1

    # Now create the variables
    var_list = [array2, array1]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"'only_creation' given '{array1}' already exists."),
    ):
        translation_driver.create_jax_var_list(
            var_list,
            only_creation=True,
        )
    assert len(translation_driver.arrays) == 1
    assert translation_driver.map_jax_var_to_sdfg(array1) == "a"


def test_driver_variable_alloc_list_handle_literal(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests part of the `JaxprTranslationDriver::create_jax_var_list()` api.

    It will test the `handle_literals` flag.
    """
    # First we have to build a jax literal.
    import numpy as np
    from jax import core as jcore

    val = np.array(1)
    aval = jcore.get_aval(val)
    lit = jcore.Literal(val, aval)
    var_list = [lit]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape("Encountered a literal but `handle_literals` was `False`."),
    ):
        translation_driver.create_jax_var_list(
            var_list,
            handle_literals=False,
        )
    assert len(translation_driver.arrays) == 0

    name_list = translation_driver.create_jax_var_list(
        var_list,
        handle_literals=True,
    )
    assert len(translation_driver.arrays) == 0
    assert name_list == [None]


def test_driver_constants(
    translation_driver: translator.JaxprTranslationDriver,
) -> None:
    """Tests part of the `JaxprTranslationDriver::_create_constants()` api.

    See also the `test_subtranslators_alu.py::test_add3` test.
    """
    import jax

    # Create the Jaxpr that we need.
    constant = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    jaxpr = jax.make_jaxpr(lambda A: A + jax.numpy.array(constant))(1.0)

    # We have to manually allocate the driver context.
    #  You should not do that.
    translation_driver._allocate_translation_ctx(name="Manual_test")

    # No create the constants.
    translation_driver._create_constants(jaxpr)

    # Test if it was created with the correct value.
    assert len(translation_driver.arrays) == 1
    assert len(translation_driver._jax_name_map) == 1
    assert next(iter(translation_driver._jax_name_map.values())) == "__const_a"
    assert len(translation_driver.sdfg.constants) == 1
    assert np.all(translation_driver.sdfg.constants["__const_a"] == constant)


def test_driver_jace_var() -> None:
    """Simple tests about the `JaCeVar` objects."""
    for iname in ["do", "", "_ _", "9al", "_!"]:
        with pytest.raises(
            expected_exception=ValueError,
            match=re.escape(f"Supplied the invalid name '{iname}'."),
        ):
            _ = JaCeVar((), dace.int8, name=iname)
