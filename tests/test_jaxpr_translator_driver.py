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


def test_driver_alloc():
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


def test_driver_fork():
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
    dolly_only_res_names = ["c"]  # reserved names that are only known to dolly
    dolly_full_res_names = org_res_names.union(dolly_only_res_names)
    dolly._allocate_translation_ctx("dolly", reserved_names=dolly_only_res_names)

    assert dolly.is_allocated()
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
    assert driver._rev_manager._next_revision == dolly_rev


def test_driver_append_state(alloc_driver):
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


if __name__ == "__main__":
    test_driver_alloc()
    test_driver_fork()
    test_driver_append_state(alloc_driver())
