# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing all translators related to arithmetic logical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from typing_extensions import override

from jace import translator
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class ALUTranslator(mapped_base.MappedOperationTranslatorBase):
    """Translator for all arithmetic and logical operations.

    The class uses `MappedOperationBaseTranslator` for generating the maps.
    Its `write_tasklet_code()` function will perform replace all literals.
    """

    __slots__ = ("_tskl_tmpl",)

    def __init__(
        self,
        prim_name: str,
        tskl_tmpl: str,
    ) -> None:
        """Initialize a base translator for primitive `prim_name` with template `tskl_tmpl`.

        Args:
            prim_name:      The name of the primitive that should be handled.
            tskl_tmpl:      Template used for generating the Tasklet code.
        """
        super().__init__(primitive_name=prim_name)
        self._tskl_tmpl = tskl_tmpl

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        """Returns the code for the Tasklet, with all parameters replaced."""
        tskl_code = self._tskl_tmpl
        if len(eqn.params) != 0:
            tskl_code = tskl_code.format(**eqn.params)
        return tskl_code


# Contains all the templates for ALU operations.
#  TODO(phimuell): Import them also from `frontend/python/replacements.py`, however, the names
#   do not fully matches the Jax names, `grep -P '^[a-zA-Z0-9_]+_p[[:space:]]+' -r -o -h | sort -u`
# fmt: off
_ALU_OPS_TMPL: Final[dict[str, str]] = {
    # Unary operations
    "pos": "__out = +(__in0)",
    "neg": "__out = -(__in0)",
    "not": "__out = not (__in0)",
    "floor": "__out = floor(__in0)",
    "ceil": "__out = ceil(__in0)",
    "round": "__out = round(__in0)",
    "abs": "__out = abs(__in0)",
    "sign": "__out = sign(__in0)",
    "sqrt": "__out = sqrt(__in0)",
    "log": "__out = log(__in0)",
    "exp": "__out = exp(__in0)",
    "integer_pow": "__out = (__in0)**({y})",  # 'y' is a parameter of the primitive
    "sin": "__out = sin(__in0)",
    "asin": "__out = asin(__in0)",
    "cos": "__out = cos(__in0)",
    "acos": "__out = acos(__in0)",
    "tan": "__out = tan(__in0)",
    "atan": "__out = atan(__in0)",
    "tanh": "__out = tanh(__in0)",

    # Binary operations
    "add": "__out = (__in0)+(__in1)",
    "add_any": "__out = (__in0)+(__in1)",  # No idea what makes `add_any` differ from `add`
    "sub": "__out = (__in0)-(__in1)",
    "mul": "__out = (__in0)*(__in1)",
    "div": "__out = (__in0)/(__in1)",
    "rem": "__out = (__in0)%(__in1)",
    "and": "__out = (__in0) and (__in1)",
    "or": "__out = (__in0) or  (__in1)",
    "pow": "__out = (__in0)**(__in1)",
    "ipow": "__out = (__in0)**(int(__in1))",
    "min": "__out = min(__in0, __in1)",
    "max": "__out = max(__in0, __in1)",
    "eq": "__out = __in0 == __in1",
    "ne": "__out = __in0 != __in1",
    "ge": "__out = __in0 >= __in1",
    "gt": "__out = __in0 > __in1",
    "le": "__out = __in0 <= __in1",
    "lt": "__out = __in0 < __in1",
}

# Create the ALU translators
for pname, ptmpl in _ALU_OPS_TMPL.items():
    translator.register_primitive_translator(ALUTranslator(pname, ptmpl))
