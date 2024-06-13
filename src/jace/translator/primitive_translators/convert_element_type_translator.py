# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the translator for the `convert_element_type` primitive."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dace
from typing_extensions import override

from jace import translator, util
from jace.translator import mapped_operation_base_translator as mapped_base


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class ConvertElementTypeTranslator(mapped_base.MappedOperationTranslatorBase):
    """
    Implements the `convert_element_type` primitive.

    Copies the input to the output and performs type conversion.

    Notes:
        This translator ignores the `new_dtype` and `weak_type` parameters of
        the equation and only performs the casting based on the type of the fields.
    """

    def __init__(self) -> None:
        super().__init__(primitive_name="convert_element_type")

    @override
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        if in_var_names[0] is None:
            raise NotImplementedError("'convert_element_type' is not supported for literals.")

        in_dtype = util.get_jax_var_dtype(eqn.invars[0]).type
        in_dtype_s: str = in_dtype.__name__
        out_dtype = util.get_jax_var_dtype(eqn.outvars[0]).type
        out_dtype_s: str = out_dtype.__name__

        # This is the base of the template that we use for conversion. You should notice
        #  that the Tasklet `__out = __in0` will fail, see commit `f5aabc3` of the
        #  prototype. Thus we have to do it in this way.
        conv_code = "__in0"

        if in_dtype == out_dtype:
            # For some reason Jax sometimes adds conversions where no are needed. In
            #  these cases we explicitly create a copy Tasklet, which is trivial and can
            #  be removed by DaCe.
            # TODO(phimuell): Create a Memlet instead.
            return f"__out = {conv_code}"

        if in_dtype_s.startswith("bool"):
            # Interestingly `__out = int(__in0)` will not work.
            conv_code = f"(1 if {conv_code} else 0)"
        if out_dtype_s.startswith("bool"):
            conv_code = f"dace.bool_({conv_code})"
        elif hasattr(dace.dtypes, out_dtype_s):
            conv_code = f"dace.{out_dtype_s}({conv_code})"
        else:
            raise NotImplementedError(
                f"Cannot convert '{in_dtype}' to '{out_dtype}' as this type is not known to DaCe."
            )
        return f"__out = {conv_code}"


_ = translator.register_primitive_translator(ConvertElementTypeTranslator())
