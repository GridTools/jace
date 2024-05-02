# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains the translation context for the `JaxprTranslationDriver`."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING

import dace
from jax import core as jcore

from jace import util as jutil


if TYPE_CHECKING:
    from jace import translator as jtrans


class _TranslationContext:
    """Represents the context of a `JaxprTranslationDriver`.

    Essentially it contains the following variables:
    - `sdfg`:
        The SDFG object that is under construction.
    - `start_state`:
        The first state in the SDFG state machine.
    - `terminal_state`:
        The current terminal state of the SDFG state machine.
    - `jax_name_map`:
        A `dict` that maps every Jax variable to its corresponding SDFG variable _name_.
    - `inp_names`:
        A `list` of the SDFG variable names that are used for input.
        Their order is the same as in `Jaxpr.invars`.
        Filled at the very beginning.
    - `out_names`:
        A `list` of the SDFG variables names that are used for output,
        Their order is the same as in `Jaxpr.outvars`.
        Only filled at the very end.
    - `rev_idx`:
        The revision index (used to generate unique names in the translation.

    Notes:
        It might be that a name appears in both the `inp_names` and `out_names` list.
            This happens if the corresponding variable is used as both input and output.
            In Jax this is called argument donation.
        This class is similar to but different to `TranslatedJaxprSDFG`.
            This class is used to represent the dynamic state of the translation object,
            `TranslatedJaxprSDFG` is used to result the end.
    """

    __slots__ = (
        "_sdfg",
        "_start_state",
        "_terminal_state",
        "_jax_name_map",
        "_inp_names",
        "_out_names",
        "_rev_idx",
    )

    def __init__(
        self,
        rev_idx: int,
        name: str | None = None,
    ) -> None:
        """Initializes the context.

        Args:
            rev_idx:    The revision index of the context.
            name:       Name of the SDFG object.
        """

        self._sdfg: dace.SDFG = dace.SDFG(name=(name or f"unnamed_SDFG_{id(self)}"))
        self._start_state: dace.SDFGState = self._sdfg.add_state(
            label="initial_state", is_start_block=True
        )
        self._terminal_state: dace.SDFGState = self._start_state
        self._jax_name_map: MutableMapping[jcore.Var | jutil.JaCeVar, str] = {}
        self._inp_names: tuple[str, ...] = ()
        self._out_names: tuple[str, ...] = ()
        self._rev_idx: int = rev_idx

    def to_translated_jaxpr_sdfg(self) -> jtrans.TranslatedJaxprSDFG:
        """Transforms `self` into a `TranslatedJaxprSDFG`."""
        return jtrans.TranslatedJaxprSDFG(
            sdfg=self._sdfg,
            start_state=self._start_state,
            terminal_state=self._terminal_state,
            jax_name_map=self._jax_name_map,
            inp_names=self._inp_names,
            out_names=self._out_names,
        )

    @property
    def sdfg(self) -> dace.SDFG:
        return self._sdfg

    @property
    def start_state(self) -> dace.SDFGState:
        return self._start_state

    @property
    def terminal_state(self) -> dace.SDFGState:
        return self._terminal_state

    @terminal_state.setter
    def terminal_state(
        self,
        new_term_state: dace.SDFGState,
    ) -> None:
        self._terminal_state = new_term_state

    @property
    def jax_name_map(self) -> MutableMapping[jcore.Var | jutil.JaCeVar, str]:
        return self._jax_name_map

    @property
    def inp_names(self) -> tuple[str, ...]:
        return self._inp_names

    @inp_names.setter
    def inp_names(
        self,
        inp_names: Sequence[str],
    ) -> None:
        if isinstance(inp_names, str):
            self._inp_names = (inp_names,)
        elif isinstance(inp_names, tuple):
            self._inp_names = inp_names
        else:
            self._inp_names = tuple(inp_names)

    @property
    def out_names(self) -> tuple[str, ...]:
        return self._out_names

    @out_names.setter
    def out_names(
        self,
        out_names: Sequence[str],
    ) -> None:
        if isinstance(out_names, str):
            self._out_names = (out_names,)
        elif isinstance(out_names, tuple):
            self._out_names = out_names
        else:
            self._out_names = tuple(out_names)

    @property
    def rev_idx(self) -> int:
        return self._rev_idx
