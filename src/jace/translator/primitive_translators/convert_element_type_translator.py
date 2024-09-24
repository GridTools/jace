# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive translator for type casting operations."""

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

    The primitive is implemented as a copy operation. However, the tasklet body
    will perform the type conversion operation.

    Note:
        The type to cast to id inferred from the output variable and the `new_dtype`
        parameter of the equation is ignored.
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

        if in_dtype == out_dtype:
            # JAX sometimes adds conversions which are not needed. In these cases
            #  make a copy out of it.
            # TODO(phimuell): Create a Memlet instead.
            return "__out = __in0"

        # A simple copy tasklet `__out = __in0` and rely on the implicit type
        #  conversion of the C++ compiler, is not enough. Due to a bug in DaCe
        #  (see https://github.com/spcl/dace/issues/1665) this conversion might be
        #  lost, thus we have to perform the conversion explicitly in the tasklet.
        conv_code = "__in0"

        if in_dtype_s.startswith("bool"):
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
