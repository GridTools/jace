# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements `select_n`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace
from typing_extensions import override

from jace import translator, util
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class SelectNTranslator(mapped_base.MappedOperationTranslatorBase):
    """Implements the `select_n` primitive, which is a generalization of `np.where`

    While `numpy.where` only supports two cases, the Jax primitive supports an arbitrary number
    of cases. In that sense it is essentially a `C` `switch` statement, only that all cases have
    to materialize.

    The behaviour is undefined if the predicate is out of bound.

    Note:
        For a better understanding this function renames its input connectors. The first one,
        which is the predicate, is renamed to `__cond` and the others are renamed again to
        `__in{i}`, starting with zero.
    """

    def __init__(self) -> None:
        super().__init__(primitive_name="select_n")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        """Writes the selection code."""
        if len(in_var_names) == 3:
            # This order is correct, since `False` is interpreted as `0`, which means the first
            #  case. DaCe seems to have some problems with bools and integer casting around,
            #  so we handle the bool case explicitly here; See also `ConvertElementTypeTranslator`.
            return "__out = __in1 if __cond else __in0"

        return "\n".join(
            ["if __cond == 0:  __out = __in0"]
            + [f"elif __cond == {i}: __out = __in{i}" for i in range(1, len(in_var_names) - 1)]
        )

    @override
    def make_input_memlets(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> dict[str, dace.Memlet]:
        """We have to add the offsets to the Memlet accesses."""
        return {
            f"__in{i-1}" if i else "__cond": dace.Memlet.simple(
                in_var_name,
                ", ".join(f"{it_idx}" for it_idx, _ in tskl_ranges),
            )
            for i, in_var_name in enumerate(in_var_names)
            if in_var_name
        }

    def literal_substitution(
        self,
        tskl_code: str,
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        """Can not be done by the base because of the renaming."""
        for i, in_var_name in enumerate(in_var_names[1:]):
            if in_var_name is not None:
                continue
            t_val = util.get_jax_literal_value(eqn.invars[i + 1])
            tskl_code = tskl_code.replace(f"__in{i}", str(t_val))
        return tskl_code


translator.register_primitive_translator(SelectNTranslator())
