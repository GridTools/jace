# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements some tests of the subtranslator builder."""

from __future__ import annotations

import re

import dace
import jax
import numpy as np
import pytest
from dace.data import Array
from jax import core as jax_core

import jace
from jace import translator, util
from jace.util import JaCeVar


# These are some JaCe variables that we use inside the tests
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
def translation_builder():
    """Returns an allocated builder instance."""
    name = "fixture_builder"
    builder = translator.JaxprTranslationBuilder(
        primitive_translators=translator.get_registered_primitive_translators()
    )
    builder._allocate_translation_ctx(name=name)
    return builder


def test_builder_alloc() -> None:
    """Tests the state right after allocation.

    Does not use the fixture because it does it on its own.
    """
    builder = translator.JaxprTranslationBuilder(
        primitive_translators=translator.get_registered_primitive_translators()
    )
    assert not builder.is_allocated(), "Builder was created allocated."
    assert len(builder._ctx_stack) == 0

    # The reserved names will be tested in `test_builder_fork()`.
    sdfg_name = "qwertzuiopasdfghjkl"
    builder._allocate_translation_ctx(name=sdfg_name)
    assert len(builder._ctx_stack) == 1
    assert builder.is_root_translator()

    sdfg: dace.SDFG = builder.sdfg

    assert builder._ctx.sdfg is sdfg
    assert builder.sdfg.name == sdfg_name
    assert sdfg.number_of_nodes() == 1
    assert sdfg.number_of_edges() == 0
    assert sdfg.start_block is builder._ctx.start_state
    assert builder._terminal_sdfg_state is builder._ctx.start_state


def test_builder_variable_alloc_auto_naming(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests simple variable allocation."""
    for i, var in enumerate([array1, array2, scal1, array3, scal2, scal3]):
        sdfg_name = translation_builder.add_array(var, update_var_mapping=True)
        sdfg_var = translation_builder.get_array(sdfg_name)
        assert sdfg_name == chr(97 + i)
        assert isinstance(sdfg_var, Array)  # Everything is now an array
        assert sdfg_var.shape == ((1,) if var.shape == () else var.shape)
        assert sdfg_var.dtype == var.dtype


def test_builder_variable_alloc_mixed_naming(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests the naming in a mixed setting.

    If `update_var_mapping=True` is given, then the naming will skip variables,
    see also `test_builder_variable_alloc_mixed_naming2()`.
    """
    #                        *       b       c       d      *      f      g
    for i, var in enumerate([narray, array1, array2, scal1, nscal, scal2, scal3]):
        sdfg_name = translation_builder.add_array(var, update_var_mapping=True)
        sdfg_var = translation_builder.get_array(sdfg_name)
        if var.name is None:
            assert sdfg_name == chr(97 + i)
        else:
            assert sdfg_name == var.name
        assert isinstance(sdfg_var, Array)  # Everything is now an array
        assert sdfg_var.shape == ((1,) if var.shape == () else var.shape)
        assert sdfg_var.dtype == var.dtype


def test_builder_variable_alloc_mixed_naming2(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests the naming in a mixed setting.

    This time we do not use `update_var_mapping=True`, instead it now depends on the
    name. This means that automatic naming will now again include all, letters, but not
    in a linear order.
    """
    letoff = 0
    #             *      a        b       c      *      d     e
    for var in [narray, array1, array2, scal1, nscal, scal2, scal3]:
        sdfg_name = translation_builder.add_array(var, update_var_mapping=var.name is None)
        sdfg_var = translation_builder.get_array(sdfg_name)
        if var.name is None:
            assert sdfg_name == chr(97 + letoff)
            letoff += 1
        else:
            assert sdfg_name == var.name
        assert isinstance(sdfg_var, Array)  # Everything is now an array
        assert sdfg_var.shape == ((1,) if var.shape == () else var.shape)
        assert sdfg_var.dtype == var.dtype


def test_builder_variable_alloc_prefix_naming(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Using the prefix to name variables."""
    prefix_1 = "__my_special_prefix"
    exp_name_1 = prefix_1 + "a"
    sdfg_name_1 = translation_builder.add_array(
        array1, name_prefix=prefix_1, update_var_mapping=False
    )
    assert exp_name_1 == sdfg_name_1

    # Because `update_var_mapping` is `False` above, 'a' will be reused.
    prefix_2 = "__my_special_prefix_second_"
    exp_name_2 = prefix_2 + "a"
    sdfg_name_2 = translation_builder.add_array(
        array1, name_prefix=prefix_2, update_var_mapping=False
    )
    assert exp_name_2 == sdfg_name_2

    # Now we use a named variables, which are also affected.
    prefix_3 = "__my_special_prefix_third_named_"
    exp_name_3 = prefix_3 + nscal.name  # type: ignore[operator]  # `.name` is not `None`.
    sdfg_name_3 = translation_builder.add_array(
        nscal, name_prefix=prefix_3, update_var_mapping=False
    )
    assert exp_name_3 == sdfg_name_3


def test_builder_variable_alloc_auto_naming_wrapped(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests the variable naming if we have more than 26 variables."""
    single_letters = [chr(x) for x in range(97, 123)]
    i = 0
    for let1 in ["", *single_letters[1:]]:  # Note `z` is followed by `ba` and not by `aa`.
        for let2 in single_letters:
            i += 1
            # Create a variable and enter it into the variable naming.
            var = JaCeVar(shape=(19, 19), dtype=dace.float64)
            sdfg_name = translation_builder.add_array(arg=var, update_var_mapping=True)
            mapped_name = translation_builder.map_jax_var_to_sdfg(var)
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


def test_builder_nested(translation_builder: translator.JaxprTranslationBuilder) -> None:
    """Tests the ability of the nesting of the builder."""

    # Now add a variable to the current subtext.
    name_1 = translation_builder.add_array(array1, update_var_mapping=True)
    assert name_1 == "a"
    assert translation_builder.map_jax_var_to_sdfg(array1) == name_1

    # For the sake of doing it add a new state to the SDFG.
    translation_builder.append_new_state("sake_state")
    assert translation_builder.sdfg.number_of_nodes() == 2
    assert translation_builder.sdfg.number_of_edges() == 1

    # Now we go one subcontext deeper
    translation_builder._allocate_translation_ctx("builder")
    assert len(translation_builder._ctx_stack) == 2
    assert translation_builder.sdfg.name == "builder"
    assert translation_builder.sdfg.number_of_nodes() == 1
    assert translation_builder.sdfg.number_of_edges() == 0
    assert not translation_builder.is_root_translator()

    # Because we have a new SDFG the mapping to previous SDFG does not work,
    #  regardless the fact that it still exists.
    with pytest.raises(
        expected_exception=KeyError,
        match=re.escape(
            f"Jax variable '{array1}' was supposed to map to '{name_1}', but no such SDFG variable is known."
        ),
    ):
        _ = translation_builder.map_jax_var_to_sdfg(array1)

    # Because the SDFGs are distinct it is possible to add `array1` to the nested one.
    #  However, it is not able to update the mapping.
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"Cannot change the mapping of '{array1}' from '{name_1}' to '{name_1}'."),
    ):
        _ = translation_builder.add_array(array1, update_var_mapping=True)
    assert name_1 not in translation_builder.sdfg.arrays

    # Without updating the mapping it is possible create the variable.
    assert name_1 == translation_builder.add_array(array1, update_var_mapping=False)

    # Now add a new variable, the map is shared, so a new name will be generated.
    name_2 = translation_builder.add_array(array2, update_var_mapping=True)
    assert name_2 == "b"
    assert name_2 == translation_builder.map_jax_var_to_sdfg(array2)

    # Now we go one stack level back.
    translation_builder._clear_translation_ctx()
    assert len(translation_builder._ctx_stack) == 1
    assert translation_builder.sdfg.number_of_nodes() == 2
    assert translation_builder.sdfg.number_of_edges() == 1

    # Again the variable that was declared in the last stack is now no longer present.
    #  Note if the nested SDFG was integrated into the parent SDFG it would be
    #  accessible
    with pytest.raises(
        expected_exception=KeyError,
        match=re.escape(
            f"Jax variable '{array2}' was supposed to map to '{name_2}', but no such SDFG variable is known."
        ),
    ):
        _ = translation_builder.map_jax_var_to_sdfg(array2)
    assert name_2 == translation_builder._jax_name_map[array2]

    # Now add a new variable, since the map is shared, we will now get the next name.
    name_3 = translation_builder.add_array(array3, update_var_mapping=True)
    assert name_3 == "c"
    assert name_3 == translation_builder.map_jax_var_to_sdfg(array3)


def test_builder_append_state(translation_builder: translator.JaxprTranslationBuilder) -> None:
    """Tests the functionality of appending states."""
    sdfg: dace.SDFG = translation_builder.sdfg

    terminal_state_1: dace.SDFGState = translation_builder.append_new_state("terminal_state_1")
    assert sdfg.number_of_nodes() == 2
    assert sdfg.number_of_edges() == 1
    assert terminal_state_1 is translation_builder._terminal_sdfg_state
    assert translation_builder._terminal_sdfg_state is translation_builder._ctx.terminal_state
    assert translation_builder._ctx.start_state is sdfg.start_block
    assert translation_builder._ctx.start_state is not terminal_state_1
    assert next(iter(sdfg.edges())).src is sdfg.start_block
    assert next(iter(sdfg.edges())).dst is terminal_state_1

    # Specifying an explicit append state that is the terminal should also update the
    #  terminal state of the builder.
    terminal_state_2: dace.SDFGState = translation_builder.append_new_state(
        "terminal_state_2", prev_state=terminal_state_1
    )
    assert sdfg.number_of_nodes() == 3
    assert sdfg.number_of_edges() == 2
    assert terminal_state_2 is translation_builder._terminal_sdfg_state
    assert sdfg.out_degree(terminal_state_1) == 1
    assert sdfg.out_degree(terminal_state_2) == 0
    assert sdfg.in_degree(terminal_state_2) == 1
    assert next(iter(sdfg.in_edges(terminal_state_2))).src is terminal_state_1

    # Specifying a previous node that is not the terminal state should not do anything.
    non_terminal_state: dace.SDFGState = translation_builder.append_new_state(
        "non_terminal_state", prev_state=terminal_state_1
    )
    assert translation_builder._terminal_sdfg_state is not non_terminal_state
    assert sdfg.in_degree(non_terminal_state) == 1
    assert sdfg.out_degree(non_terminal_state) == 0
    assert next(iter(sdfg.in_edges(non_terminal_state))).src is terminal_state_1


def test_builder_variable_multiple_variables(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Add an already known variable, but with a different name."""
    # Now we will add `array1` and then different ways of updating it.
    narray1: str = translation_builder.add_array(array1, update_var_mapping=True)

    # It will fail if we use the prefix, because we also want to update.
    prefix = "__jace_prefix"
    prefix_expected_name = prefix + narray1
    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            f"Cannot change the mapping of '{array1}' from '{translation_builder.map_jax_var_to_sdfg(array1)}' to '{prefix_expected_name}'."
        ),
    ):
        _ = translation_builder.add_array(array1, update_var_mapping=True, name_prefix=prefix)
    assert prefix_expected_name not in translation_builder.sdfg.arrays

    # But if we do not want to update it then it works.
    prefix_sdfg_name = translation_builder.add_array(
        array1, update_var_mapping=False, name_prefix=prefix
    )
    assert prefix_expected_name == prefix_sdfg_name
    assert prefix_expected_name in translation_builder.sdfg.arrays
    assert narray1 == translation_builder.map_jax_var_to_sdfg(array1)


def test_builder_variable_invalid_prefix(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Use invalid prefix."""
    # It will fail if we use the prefix, because we also want to update.
    for iprefix in ["0_", "_ja ", "_!"]:
        with pytest.raises(
            expected_exception=ValueError,
            match=re.escape(f"add_array({array1}): The proposed name '{iprefix}a', is invalid."),
        ):
            _ = translation_builder.add_array(array1, update_var_mapping=False, name_prefix=iprefix)
        assert len(translation_builder.sdfg.arrays) == 0


def test_builder_variable_alloc_list(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests part of the `JaxprTranslationBuilder.create_jax_var_list()` api."""
    var_list_1 = [array1, nscal, scal2]
    exp_names_1 = ["a", nscal.name, "c"]

    res_names_1 = translation_builder.create_jax_var_list(var_list_1, update_var_mapping=True)
    assert len(translation_builder.arrays) == 3
    assert res_names_1 == exp_names_1

    # Now a mixture of the collection and creation.
    var_list_2 = [array2, nscal, scal1]
    exp_names_2 = ["d", nscal.name, "e"]

    res_names_2 = translation_builder.create_jax_var_list(var_list_2, update_var_mapping=True)
    assert res_names_2 == exp_names_2
    assert len(translation_builder.arrays) == 5


@pytest.mark.skip(reason="'create_jax_var_list()' does not clean up in case of an error.")
def test_builder_variable_alloc_list_cleaning(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests part of the `JaxprTranslationBuilder.create_jax_var_list()` api.

    It will fail because `update_var_mapping=False` thus the third variable will
    cause an error because it is proposed to `a`, which is already used.
    """
    var_list = [array1, nscal, scal2]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"add_array({scal2}): The proposed name 'a', is used."),
    ):
        _ = translation_builder.create_jax_var_list(var_list)

    assert len(translation_builder.arrays) == 0


def test_builder_variable_alloc_list_prevent_creation(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests part of the `JaxprTranslationBuilder.create_jax_var_list()` api.

    It will test the `prevent_creation` flag.
    """
    # First create a variable.
    translation_builder.add_array(array1, update_var_mapping=True)
    assert len(translation_builder.arrays) == 1

    # Now create the variables
    var_list = [array1, array2]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"'prevent_creation' given but have to create '{array2}'."),
    ):
        translation_builder.create_jax_var_list(var_list, prevent_creation=True)
    assert len(translation_builder.arrays) == 1
    assert translation_builder.map_jax_var_to_sdfg(array1) == "a"


@pytest.mark.skip(reason="'create_jax_var_list()' does not clean up in case of an error.")
def test_builder_variable_alloc_list_only_creation(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests part of the `JaxprTranslationBuilder.create_jax_var_list()` api.

    It will test the `only_creation` flag.
    """
    # First create a variable.
    translation_builder.add_array(array1, update_var_mapping=True)
    assert len(translation_builder.arrays) == 1

    # Now create the variables
    var_list = [array2, array1]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(f"'only_creation' given '{array1}' already exists."),
    ):
        translation_builder.create_jax_var_list(var_list, only_creation=True)
    assert len(translation_builder.arrays) == 1
    assert translation_builder.map_jax_var_to_sdfg(array1) == "a"


def test_builder_variable_alloc_list_handle_literal(
    translation_builder: translator.JaxprTranslationBuilder,
) -> None:
    """Tests part of the `JaxprTranslationBuilder.create_jax_var_list()` api.

    It will test the `handle_literals` flag.
    """

    val = np.array(1)
    aval = jax_core.get_aval(val)
    lit = jax_core.Literal(val, aval)
    var_list = [lit]

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape("Encountered a literal but `handle_literals` was `False`."),
    ):
        translation_builder.create_jax_var_list(var_list, handle_literals=False)
    assert len(translation_builder.arrays) == 0

    name_list = translation_builder.create_jax_var_list(var_list, handle_literals=True)
    assert len(translation_builder.arrays) == 0
    assert name_list == [None]


def test_builder_constants(translation_builder: translator.JaxprTranslationBuilder) -> None:
    """Tests part of the `JaxprTranslationBuilder._create_constants()` api.

    See also the `test_subtranslators_alu.py::test_add3` test.
    """
    # Create the Jaxpr that we need.
    constant = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    jaxpr = jax.make_jaxpr(lambda A: A + jax.numpy.array(constant))(1.0)

    # We have to manually allocate the builder context.
    #  You should not do that.
    translation_builder._allocate_translation_ctx(name="Manual_test")

    # No create the constants.
    translation_builder._create_constants(jaxpr)

    # Test if it was created with the correct value.
    assert len(translation_builder.arrays) == 1
    assert len(translation_builder._jax_name_map) == 1
    assert next(iter(translation_builder._jax_name_map.values())) == "__const_a"
    assert len(translation_builder.sdfg.constants) == 1
    assert np.all(translation_builder.sdfg.constants["__const_a"] == constant)


def test_builder_scalar_return_value() -> None:
    """Tests if scalars can be returned directly."""

    def scalar_ops(A: float) -> float:
        return A + A - A * A

    lower_cnt = [0]

    @jace.jit
    def wrapped(A: float) -> float:
        lower_cnt[0] += 1
        return scalar_ops(A)

    vals = np.random.random(100)  # noqa: NPY002
    for i in range(vals.size):
        res = wrapped(vals[i])
        ref = scalar_ops(vals[i])
        assert np.allclose(res, ref)
    assert lower_cnt[0] == 1


@pytest.mark.skip(reason="Currently 'scalar' return values, are actually shape '(1,)' arrays.")
def test_builder_scalar_return_type() -> None:
    """Tests if the type is the same, in case of scalar return."""

    @jace.jit
    def wrapped(A: np.float64) -> np.float64:
        return A + A - A * A

    A = np.float64(1.0)
    assert type(A) is np.float64, f"Expected type 'np.float64', but got '{type(A).__name__}'."


def test_builder_jace_var() -> None:
    """Simple tests about the `JaCeVar` objects."""
    for iname in ["do", "", "_ _", "9al", "_!"]:
        with pytest.raises(
            expected_exception=ValueError, match=re.escape(f"Supplied the invalid name '{iname}'.")
        ):
            _ = JaCeVar((), dace.int8, name=iname)


def test_builder_F_strides() -> None:
    """Tests if we can lower without a standard stride.

    Notes:
        This tests if the restriction is currently in place.
        See also `tests/test_caching.py::test_caching_strides`.
    """

    @jace.jit
    def testee(A: np.ndarray) -> np.ndarray:
        return A + 10.0

    F = np.full((4, 3), 10, dtype=np.float64, order="F")

    with pytest.raises(
        expected_exception=NotImplementedError,
        match=re.escape("Currently can not yet handle strides beside 'C_CONTIGUOUS'."),
    ):
        _ = testee(F)
