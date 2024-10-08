# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Primitive translators related to all arithmetic, logical and comparison operations.

Todo:
    - Hijack Jax to inject a proper modulo operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import dace
from typing_extensions import override

from jace import translator, util
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class ArithmeticOperationTranslator(mapped_base.MappedOperationTranslatorBase):
    """
    Translator for all arithmetic operations and comparisons.

    Args:
        prim_name: The name of the primitive that should be handled.
        tskl_tmpl: Template used for generating the Tasklet code.

    Note:
        Logical and bitwise operations are implemented by `LogicalOperationTranslator`.
    """

    def __init__(self, prim_name: str, tskl_tmpl: str) -> None:
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
        return self._tskl_tmpl.format(**eqn.params)


class LogicalOperationTranslator(mapped_base.MappedOperationTranslatorBase):
    """
    Translator for all logical operations.

    The reason why the logical operations are separated from the arithmetic
    operations is quite complicated and in fact the whole thing is harder than
    it should be. NumPy has two kinds of these operations, i.e.
    `logical_{and, or, xor, not}()` and `bitwise_{and, or, xor, not}()`, but Jax
    has only a single kind of logical operation, that operate in bitwise mode.
    The first idea would be to use `ArithmeticOperationTranslator` with a template
    such as `__out = __in0 & __in1` or `__out = ~__in0`. Since DaCe eventually
    generates C++ code and C++ has a native bool type, and `true` is guaranteed
    to be `1` and `false` equals `0`, this works for all operations except `not`,
    as `~true` in C++ is essentially `~1`, which is again `true`!
    Thus the `not` primitive must be handled separately.

    The solution to the problem is to introduce two templates, one used for the
    bool context and one used in the integer context. This works because depending
    if the `logical_*()` or `bitwise_*()` functions are used the input is either
    of type bool or an integer.

    Args:
        prim_name: The name of the primitive that should be handled.
        bitwise_tmpl: The template used for the bitwise case.
        logical_tmpl: The template used for the logical case.

    Note:
        Since it does not make sense to single out `not` and keep the other
        logical operations in `ArithmeticOperationTranslator` all of them are
        handled by this class.
    """

    def __init__(self, prim_name: str, bitwise_tmpl: str, logical_tmpl: str) -> None:
        super().__init__(primitive_name=prim_name)
        self._bitwise_tmpl = bitwise_tmpl
        self._logical_tmpl = logical_tmpl

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        if all(util.get_jax_var_dtype(invar) is dace.bool_ for invar in eqn.invars):
            return self._logical_tmpl
        return self._bitwise_tmpl


# Maps the name of an arithmetic JAX primitive to the code template that is used to
#  generate the body of the mapped tasklet. These are used to instantiate the
#  `ArithmeticOperationTranslator` objects.
# fmt: off
_ARITMETIC_OPERATION_TEMPLATES: Final[dict[str, str]] = {
    # Unary operations
    "pos": "__out = +(__in0)",
    "neg": "__out = -(__in0)",

    "floor": "__out = floor(__in0)",
    "ceil": "__out = ceil(__in0)",
    "round": "__out = round(__in0)",

    "abs": "__out = abs(__in0)",
    "sign": "__out = sign(__in0)",
    "exp": "__out = exp(__in0)",
    "exp2": "__out = exp2(__in0)",
    "expm1": "__out = expm1(__in0)",
    "log": "__out = log(__in0)",
    "log1p": "__out = log1p(__in0)",
    "conj": "__out = conj(__in0)",
    "sqrt": "__out = sqrt(__in0)",
    "cbrt": "__out = cbrt(__in0)",

    "integer_pow": "__out = (__in0)**({y})",  # 'y' is a parameter of the primitive
    "is_finite": "__out = isfinite(__in0)",

    "sin": "__out = sin(__in0)",
    "asin": "__out = asin(__in0)",
    "cos": "__out = cos(__in0)",
    "acos": "__out = acos(__in0)",
    "tan": "__out = tan(__in0)",
    "atan": "__out = atan(__in0)",

    "sinh": "__out = sinh(__in0)",
    "asinh": "__out = asinh(__in0)",
    "cosh": "__out = cosh(__in0)",
    "acosh": "__out = acosh(__in0)",
    "tanh": "__out = tanh(__in0)",
    "atanh": "__out = atanh(__in0)",

    # Binary operations
    "add": "__out = (__in0)+(__in1)",
    "add_any": "__out = (__in0)+(__in1)",  # No idea what makes `add_any` differ from `add`
    "sub": "__out = (__in0)-(__in1)",
    "mul": "__out = (__in0)*(__in1)",
    "div": "__out = (__in0)/(__in1)",
    "rem": "__out = (__in0)%(__in1)",
    "pow": "__out = (__in0)**(__in1)",
    "min": "__out = min((__in0), (__in1))",
    "max": "__out = max((__in0), (__in1))",

    "eq": "__out = (__in0) == (__in1)",
    "ne": "__out = (__in0) != (__in1)",
    "ge": "__out = (__in0) >= (__in1)",
    "gt": "__out = (__in0) > (__in1)",
    "le": "__out = (__in0) <= (__in1)",
    "lt": "__out = (__in0) < (__in1)",

    "atan2": "__out = atan2((__in0), (__in1))",

    "nextafter": "__out = nextafter((__in0), (__in1))",

    # Ternary operations
    "clamp": "__out = ((__in0) if (__in1) < (__in0) else ((__in1) if (__in1) < (__in2) else (__in2)))"
}


# Maps the name of a logical primitive to the two code templates, first the integer
#  case and second the boolean case, that are used to create the body of the mapped
#  tasklet. They are used to instantiate the `LogicalOperationTranslator` translators.
_LOGICAL_OPERATION_TEMPLATES: Final[dict[str, dict[str, str]]] = {
    "or": {
        "bitwise_tmpl": "__out = (__in0) | (__in1)",
        "logical_tmpl": "__out = (__in0) or (__in1)",
    },
    "not": {
        "bitwise_tmpl": "__out = ~(__in0)",
        "logical_tmpl": "__out = not (__in0)",
    },
    "and": {
        "bitwise_tmpl": "__out = (__in0) & (__in1)",
        "logical_tmpl": "__out = (__in0) and (__in1)",
    },
    "xor": {
        "bitwise_tmpl": "__out = (__in0) ^ (__in1)",
        "logical_tmpl": "__out = (__in0) != (__in1)",
    },
}
# fmt: on


# Instantiate the arithmetic and logical translators from the templates.
for pname, ptmpl in _ARITMETIC_OPERATION_TEMPLATES.items():
    translator.register_primitive_translator(ArithmeticOperationTranslator(pname, ptmpl))
for pname, ptmpl in _LOGICAL_OPERATION_TEMPLATES.items():  # type: ignore[assignment]  # Type confusion
    translator.register_primitive_translator(LogicalOperationTranslator(pname, **ptmpl))  # type: ignore[arg-type]  # Type confusion
