# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements some tests of the subtranslator driver."""

from __future__ import annotations

import dace
import pytest

from jace import translator as jtrans


@pytest.fixture(scope="module")
def alloc_driver():
    """Returns an allocated driver instance."""
    name = "fixture_driver"
    driver = jtrans.JaxprTranslationDriver()
    driver._allocate_translation_ctx(name=name)
    return driver


def test_driver_alloc() -> None:
    """Tests the state right after allocation."""
    driver = jtrans.JaxprTranslationDriver()
    assert not driver.is_allocated(), "Driver was created allocated."

    # The reserved names will be tested in `test_driver_fork()`.
    sdfg_name = "qwertzuiopasdfghjkl"
    driver._allocate_translation_ctx(name=sdfg_name)

    sdfg: dace.SDFG = driver.get_sdfg()

    assert driver.get_sdfg().name == sdfg_name
    assert sdfg.number_of_nodes() == 1
    assert sdfg.number_of_edges() == 0
    assert sdfg.start_block is driver._init_sdfg_state
    assert driver.get_terminal_sdfg_state() is driver._init_sdfg_state


def test_driver_fork() -> None:
    """Tests the fork ability of the driver."""

    # This is the parent driver.
    driver = jtrans.JaxprTranslationDriver()
    assert not driver.is_allocated(), "Driver should not be allocated."

    with pytest.raises(expected_exception=RuntimeError, match="Only allocated driver can fork."):
        _ = driver.fork()
    #

    # We allocate the driver directly, because we need to set some internals.
    #  This is also the reason why we do not use the fixture.
    org_res_names = {"a", "b"}
    driver._allocate_translation_ctx("driver", reserved_names=org_res_names)
    assert driver.is_allocated()
    assert driver._reserved_names == org_res_names

    # Now we allocate a child
    dolly = driver.fork()
    dolly_rev = dolly.get_rev_idx()
    assert not dolly.is_allocated()
    assert not dolly.is_head_translator()
    assert driver.is_head_translator()
    assert dolly.same_family(driver)
    assert driver.same_family(dolly)
    assert driver._sub_translators is dolly._sub_translators
    assert driver._rev_manager is dolly._rev_manager
    assert dolly._reserved_names == driver._reserved_names
    assert dolly._reserved_names is not driver._reserved_names

    # Test if allocation of fork works properly
    dolly_only_res_names = ["c"]  # reserved names that are only known to dolly; Added latter
    dolly_full_res_names = org_res_names.union(dolly_only_res_names)
    dolly._allocate_translation_ctx(
        "dolly",
    )

    assert dolly.is_allocated()
    assert dolly._reserved_names == org_res_names
    assert driver._reserved_names == org_res_names

    # Now adding reserved names to dolly after construction.
    dolly.add_reserved_names(None)
    assert dolly._reserved_names == org_res_names
    dolly.add_reserved_names(dolly_only_res_names)
    assert dolly._reserved_names == dolly_full_res_names
    assert driver._reserved_names == org_res_names

    # Now we deallocate dolly
    dolly._clear_translation_ctx()
    assert not dolly.is_allocated()
    assert dolly._reserved_names is not None
    assert dolly._reserved_names == dolly_full_res_names

    # Now we test if the revision index is again increased properly.
    dolly2 = driver.fork()
    assert dolly_rev < dolly2.get_rev_idx()
    assert dolly2.same_family(dolly)
    assert dolly2.same_family(driver)

    # Deallocate the driver
    driver._clear_translation_ctx()
    assert not driver.is_allocated()
    assert driver.is_head_translator()
    assert driver._reserved_names is None


def test_driver_append_state(alloc_driver: jtrans.JaxprTranslationDriver) -> None:
    """Tests the functionality of appending states."""
    sdfg: dace.SDFG = alloc_driver.get_sdfg()

    terminal_state_1: dace.SDFGState = alloc_driver.append_new_state("terminal_state_1")
    assert sdfg.number_of_nodes() == 2
    assert sdfg.number_of_edges() == 1
    assert terminal_state_1 is alloc_driver.get_terminal_sdfg_state()
    assert alloc_driver.get_terminal_sdfg_state() is alloc_driver._term_sdfg_state
    assert alloc_driver._init_sdfg_state is sdfg.start_block
    assert alloc_driver._init_sdfg_state is not terminal_state_1
    assert next(iter(sdfg.edges())).src is sdfg.start_block
    assert next(iter(sdfg.edges())).dst is terminal_state_1

    # Specifying an explicit append state that is the terminal should also update the terminal state of the driver.
    terminal_state_2: dace.SDFGState = alloc_driver.append_new_state(
        "terminal_state_2", prev_state=terminal_state_1
    )
    assert sdfg.number_of_nodes() == 3
    assert sdfg.number_of_edges() == 2
    assert terminal_state_2 is alloc_driver.get_terminal_sdfg_state()
    assert sdfg.out_degree(terminal_state_1) == 1
    assert sdfg.out_degree(terminal_state_2) == 0
    assert sdfg.in_degree(terminal_state_2) == 1
    assert next(iter(sdfg.in_edges(terminal_state_2))).src is terminal_state_1

    # Specifying a previous node that is not the terminal state should not do anything.
    non_terminal_state: dace.SDFGState = alloc_driver.append_new_state(
        "non_terminal_state", prev_state=terminal_state_1
    )
    assert alloc_driver.get_terminal_sdfg_state() is not non_terminal_state
    assert sdfg.in_degree(non_terminal_state) == 1
    assert sdfg.out_degree(non_terminal_state) == 0
    assert next(iter(sdfg.in_edges(non_terminal_state))).src is terminal_state_1


def test_driver_array(alloc_driver: jtrans.JaxprTranslationDriver) -> None:
    """This function tests the array creation routines.

    However, it does so without using Jax variables.
    """
    from dace.data import Array, Data, Scalar

    from jace.util import JaCeVar

    # Since we do not have Jax variables, we are using JaCe substitute for it.

    # Creating a scalar.
    scal1_j = JaCeVar("scal1", (), dace.float64)
    scal1_: str = alloc_driver.add_array(
        arg=scal1_j,
        update_var_mapping=True,
    )
    scal1: Data = alloc_driver.get_array(scal1_)
    assert scal1 is alloc_driver.get_array(scal1_j)
    assert scal1_ == alloc_driver.map_jax_var_to_sdfg(scal1_j)
    assert isinstance(scal1, Scalar)
    assert scal1_ == scal1_j.name
    assert scal1.dtype == scal1_j.dtype

    # Create a scalar and force it as an array
    scal2_j = JaCeVar("scal2", (), dace.int64)
    scal2_: str = alloc_driver.add_array(
        arg=scal2_j,
        force_array=True,
    )
    scal2: Data = alloc_driver.get_array(scal2_)
    assert isinstance(scal2, Array)
    assert scal2_ == scal2_j.name
    assert scal2.shape == (1,)
    assert scal2.strides == (1,)
    assert scal2.dtype == scal2_j.dtype

    # Create a scalar force it as an array and use symbolic strides.
    scal3_j = JaCeVar("scal3", (), dace.int64)
    scal3_: str = alloc_driver.add_array(
        arg=scal3_j,
        force_array=True,
        symb_strides=True,  # Will have no effect.
    )
    scal3: Data = alloc_driver.get_array(scal3_)
    assert isinstance(scal2, Array)
    assert scal3_ == scal3_j.name
    assert scal3.shape == (1,)
    assert scal3.strides == (1,)
    assert scal3.dtype == scal3_j.dtype

    # Using a special name for the variable
    scal4_j = scal3_j
    scal4_n = "scal4_special_name"
    scal4_: str = alloc_driver.add_array(
        arg=scal4_j,
        alt_name=scal4_n,
        update_var_mapping=True,
    )
    assert scal4_ == scal4_n
    assert scal4_ == alloc_driver.map_jax_var_to_sdfg(scal4_j)

    # Test the prefix functionality
    scal5_j = JaCeVar("scal5", (), dace.float64)
    scal5_p = "my_prefix"
    scal5_: str = alloc_driver.add_array(
        arg=scal5_j,
        name_prefix=scal5_p,
    )
    assert scal5_.startswith(scal5_p)
    assert scal5_j.name in scal5_

    # Allocating an array
    arr1_j = JaCeVar("arr1", (5, 3), dace.float32)
    arr1_: str = alloc_driver.add_array(
        arg=arr1_j,
    )
    arr1: Data = alloc_driver.get_array(arr1_)
    assert isinstance(arr1, Array)
    assert arr1_ == arr1_j.name
    assert arr1.shape == arr1_j.shape
    assert arr1.strides == (3, 1)
    assert arr1.dtype == arr1_j.dtype

    # Create a variable that has a name that is already known.
    arr2_j = JaCeVar(arr1_, (10,), dace.float64)
    with pytest.raises(
        expected_exception=ValueError,
        match=f"Can't create variable '{arr2_j.name}', variable is already created.",
    ):
        arr2_: str = alloc_driver.add_array(arg=arr2_j)
    with pytest.raises(expected_exception=ValueError, match=f"Variable '{arr1_}' already exists."):
        # `alt_name` will not work because variable still exists.
        arr2_ = alloc_driver.add_array(arg=arr2_j, alt_name=arr2_j.name)
    # However, specifying `find_new_name` will solve this issue
    #  NOTE: Doing this is not a good idea.
    arr2_ = alloc_driver.add_array(
        arg=arr2_j,
        find_new_name=True,
    )
    assert arr2_.startswith("_jax_variable__" + arr2_j.name)

    # Create a variable that has a custom stride
    arr3_j = JaCeVar("arr3", (5, 1, 3), dace.float64)
    arr3_st = (5, 3, 2)
    arr3_: str = alloc_driver.add_array(
        arg=arr3_j,
        strides=arr3_st,
    )
    arr3: Data = alloc_driver.get_array(arr3_)
    assert isinstance(arr3, Array)
    assert arr3.shape == arr3_j.shape
    assert arr3.strides == arr3_st

    # Test if specifying `symb_strides` and a stride at the same time is an error.
    arr4_j = JaCeVar("arr4", arr3_j.shape, dace.uintp)
    arr4_st = arr3_st
    with pytest.raises(
        expected_exception=ValueError,
        match="Specified 'symb_strides' and 'stride at the same time.",
    ):
        arr4_: str = alloc_driver.add_array(
            arg=arr4_j,
            symb_strides=True,
            strides=arr4_st,
        )

    # Test if specifying the symbolic stride alone works.
    #  Because a shape is `1` there should be no symbolic for it.
    arr4_ = alloc_driver.add_array(
        arg=arr4_j,
        symb_strides=True,
    )
    arr4: Data = alloc_driver.get_array(arr4_)
    assert isinstance(arr4, Array)
    assert arr4.shape == arr4_j.shape

    for shp, stri in zip(arr4.shape, arr4.strides):
        if shp == 1:
            assert isinstance(stri, int)
            assert stri == 0, f"Expected a stride of 0, but got '{stri}'."
        else:
            assert isinstance(stri, (str, dace.symbol))


if __name__ == "__main__":
    test_driver_alloc()
    test_driver_fork()
