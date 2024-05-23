# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing all translators related to arithmetic logical operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, cast

import numpy as np
from jax import core as jax_core
from typing_extensions import override

from jace import translator
from jace.translator.primitive_translators.mapped_operation_base_translator import (
    MappedOperationTranslatorBase,
)


class ALUTranslator(MappedOperationTranslatorBase):
    """Translator for all arithmetic and logical operations.

    The class uses `MappedOperationBaseTranslator` for generating the maps.
    Its `write_tasklet_code()` function will perform replace all literals.
    """

    __slots__ = "_tskl_tmpl"

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
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        """Return the code that should be put inside the Tasklet, with all parameters and literals substituted with their values.

        Args:
            in_var_names:   The list of SDFG variables used as input.
            eqn:            The equation.
        """
        tskl_code = self._tskl_tmpl
        for i, in_var_name in enumerate(in_var_names):
            if in_var_name is not None:
                continue

            jax_in_var: jax_core.Literal = cast(jax_core.Literal, eqn.invars[i])
            if jax_in_var.aval.shape == ():
                t_val = jax_in_var.val
                if isinstance(t_val, np.ndarray):
                    t_val = jax_in_var.val.max()  # I do not know a better way in that case
                tskl_code = tskl_code.replace(f"__in{i}", str(t_val))
            else:
                raise ValueError(f"Can not handle non scalar literals: {jax_in_var}")
        if len(eqn.params) != 0:
            tskl_code = tskl_code.format(**eqn.params)

        return tskl_code


# Contains all the templates for ALU operations.
# fmt: off
_ALU_OPS_TMPL: Final[dict[str, str]] = {
    # Unary operations
    "pos": "__out0 = +(__in0)",
    "neg": "__out0 = -(__in0)",
    "not": "__out0 = not (__in0)",
    "floor": "__out0 = floor(__in0)",
    "ceil": "__out0 = ceil(__in0)",
    "round": "__out0 = round(__in0)",
    "abs": "__out0 = abs(__in0)",
    "sign": "__out0 = sign(__in0)",
    "sqrt": "__out0 = sqrt(__in0)",
    "log": "__out0 = log(__in0)",
    "exp": "__out0 = exp(__in0)",
    "integer_pow": "__out0 = (__in0)**({y})",  # 'y' is a parameter of the primitive
    "sin": "__out0 = sin(__in0)",
    "asin": "__out0 = asin(__in0)",
    "cos": "__out0 = cos(__in0)",
    "acos": "__out0 = acos(__in0)",
    "tan": "__out0 = tan(__in0)",
    "atan": "__out0 = atan(__in0)",
    "tanh": "__out0 = tanh(__in0)",

    # Binary operations
    "add": "__out0 = (__in0)+(__in1)",
    "add_any": "__out0 = (__in0)+(__in1)",  # No idea what makes `add_any` differ from `add`
    "sub": "__out0 = (__in0)-(__in1)",
    "mul": "__out0 = (__in0)*(__in1)",
    "div": "__out0 = (__in0)/(__in1)",
    "rem": "__out0 = (__in0)%(__in1)",
    "and": "__out0 = (__in0) and (__in1)",
    "or": "__out0 = (__in0) or  (__in1)",
    "pow": "__out0 = (__in0)**(__in1)",
    "ipow": "__out0 = (__in0)**(int(__in1))",
    "min": "__out0 = min(__in0, __in1)",
    "max": "__out0 = max(__in0, __in1)",
    "eq": "__out0 = __in0 == __in1",
    "ne": "__out0 = __in0 != __in1",
    "ge": "__out0 = __in0 >= __in1",
    "gt": "__out0 = __in0 > __in1",
    "le": "__out0 = __in0 <= __in1",
    "lt": "__out0 = __in0 < __in1",
}

# Create the ALU translators
for pname, ptmpl in _ALU_OPS_TMPL.items():
    translator.register_primitive_translator(ALUTranslator(pname, ptmpl))
