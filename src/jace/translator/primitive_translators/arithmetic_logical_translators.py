# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing all translators related to arithmetic and logical operations.

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
    """Translator for all arithmetic operations.

    The class makes use of the `MappedOperationTranslatorBase`. It only implements
    the `write_tasklet_code()` to generate the code for a Tasklet from a template.

    Args:
        prim_name:      The name of the primitive that should be handled.
        tskl_tmpl:      Template used for generating the Tasklet code.

    Note:
        - It does not implement the logical operations, they are implemented by
            the `LogicalOperationTranslator` class.
        - It does not implement `mod` nor `fmod` as they are translated to some
            nested `pjit` implementation by Jax for unknown reasons.
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
        tskl_code = self._tskl_tmpl
        if len(eqn.params) != 0:
            tskl_code = tskl_code.format(**eqn.params)
        return tskl_code


class LogicalOperationTranslator(mapped_base.MappedOperationTranslatorBase):
    """Translator for all logical operations.

    The reason why the logical operations are separated from the arithmetic
    operation is quite complicated, and in fact the whole thing is harder than
    it should be. NumPy has two kinds of these operations, i.e.
    `logical_{and, or, xor, not}()` and `bitwise_{and, or, xor, not}()`, but Jax
    has only a single kind of logical operations, that operate in bitwise mode.
    The first idea would be to use `ArithmeticOperationTranslator` with a template
    such as `__out = __in0 & __in1` or `__out = ~__in0`. Since DaCe eventually
    generates C++ code and C++ has a native bool type, and `true` is guaranteed
    to be `1` and `false` equals `0`, this works for all operations except `not`,
    as `~true` in C++ is again `true`. Thus the `not` primitive must be handled
    separately, however, it does not make sense to split the logical operations,
    thus all of them are handled by this class.

    The solution to the problem is, to introduce two templates, one used for the
    bool context and one used in the integer context. This works because depending
    if the `logical_*()` or `bitwise_*()` functions are used the input is either
    of type bool or an integer.

    Args:
        prim_name:      The name of the primitive that should be handled.
        int_tmpl:       The template used for the integer case.
        bool_tmpl:      The template used for the bool case.

    Notes:
        This class does not do parameter substitution as the
        `ArithmeticOperationTranslator` does.
    """

    def __init__(self, prim_name: str, int_tmpl: str, bool_tmpl: str) -> None:
        super().__init__(primitive_name=prim_name)
        self._int_tmpl = int_tmpl
        self._bool_tmpl = bool_tmpl

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        if all(util.get_jax_var_dtype(invar) is dace.bool_ for invar in eqn.invars):
            return self._bool_tmpl
        return self._int_tmpl


# Contains the code templates for all supported arithmetic operations.
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
}


# Contains the code templates for all logical operations.
#  The first one is for the integer case, the second for the bool case.
_LOGICAL_OPERATION_TEMPLATES: Final[dict[str, tuple[str, str]]] = {
    "or": ("__out = (__in0) | (__in1)",  "__out = (__in0) or (__in1)"),
    "not": ("__out = ~(__in0)", "__out = not (__in0)"),
    "and": ("__out = (__in0) & (__in1)", "__out = (__in0) and (__in1)"),
    "xor": ("__out = (__in0) ^ (__in1)", "__out = (__in0) != (__in1)"),
}



# Create the arithmetic translators
for pname, ptmpl in _ARITMETIC_OPERATION_TEMPLATES.items():
    translator.register_primitive_translator(ArithmeticOperationTranslator(pname, ptmpl))

# create the logical translators.
for pname, (itmpl, btmpl) in _LOGICAL_OPERATION_TEMPLATES.items():
    translator.register_primitive_translator(LogicalOperationTranslator(pname, itmpl, btmpl))
