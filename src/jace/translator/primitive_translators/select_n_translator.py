# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator for select operations, i.e. generalized `np.where()`."""

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
    """
    Implements the `select_n` primitive.

    The `select_n` primitive is a generalization of `np.where`, that can take an
    arbitrary number of cases, which are selected by an integer predicate.
    The behaviour is undefined if the predicate is out of bound.

    Note:
        For a better understanding this function renames its input connectors.
        The first one, which is the predicate, is renamed to `__cond` and the
        others are renamed again to `__in{i}`, starting with zero.
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
        if len(in_var_names) == 3:  # noqa: PLR2004 [magic-value-comparison]  # Ternary conditional expression.
            # The order is correct, since `False` is interpreted as `0`,
            #  which means "the first case".
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
        return {
            f"__in{i - 1}" if i else "__cond": dace.Memlet.simple(
                in_var_name, ", ".join(f"{it_idx}" for it_idx, _ in tskl_ranges)
            )
            for i, in_var_name in enumerate(in_var_names)
            if in_var_name
        }

    @override
    def literal_substitution(
        self, tskl_code: str, in_var_names: Sequence[str | None], eqn: jax_core.JaxprEqn
    ) -> str:
        assert in_var_names[0]  # Condition can never be a literal.
        for i, in_var_name in enumerate(in_var_names[1:]):
            if in_var_name is None:
                t_val = util.get_jax_literal_value(eqn.invars[i + 1])
                tskl_code = tskl_code.replace(f"__in{i}", str(t_val))
        return tskl_code


translator.register_primitive_translator(SelectNTranslator())
