# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the Translator for the `convert_element_type` primitive."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import dace
from jax import core as jax_core
from typing_extensions import override

from jace import translator
from jace.translator.primitive_translators.mapped_operation_base_translator import (
    MappedOperationBaseTranslator,
)


class ConvertElementTypeTranslator(MappedOperationBaseTranslator):
    """Implements the `convert_element_type` primitive.

    Copies the input to the output and performs type conversion.

    Notes:
        This translator ignores the `new_dtype` and `weak_type` parameter of the equation and only performs casting

    Todo:
        I occasionally Jax converts from the same type to another type.
            This case should be handled by a Memlet directly, which can then be removed.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(primitive_name="convert_element_type")

    @override
    def write_tasklet_code(
        self,
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        """Return the code that should be put inside the Tasklet.

        Note that returned code is not processed any further.
        Thus the function has to apply literal removal on its own.

        Args:
            in_var_names:   The list of SDFG variables used as input.
            eqn:            The equation.
        """
        assert in_var_names[0] is not None

        in_var_name: str = in_var_names[0]
        in_dtype = eqn.invars[0].aval.dtype
        in_dtype_s: str = str(in_dtype)
        out_dtype = eqn.outvars[0].aval.dtype
        out_dtype_s: str = str(out_dtype)

        if in_var_name is None:
            raise NotImplementedError("'convert_element_type' is not supported for literals.")
        if in_dtype == out_dtype:
            # TODO(phimuell): make this into a pure Memlet such that it can be optimized away by DaCe.
            # Believe it or not but it happens.
            warnings.warn(
                "convert_element_type({eqn}): is useless, because input and output have same type.",
                stacklevel=1,  # Find a better one
            )

        # This is the base of the template that we use for conversion.
        #  You should notice that the Tasklet `__out0 = __in0` will fail, see commit `f5aabc3` of the prototype.
        #  Thus we have to do it in this way.
        conv_code = "__in0"

        if in_dtype_s.startswith("bool") and out_dtype_s.startswith("int"):
            # Interestingly `__out0 = int(__in0)` will fail, Dace will optimize it away.
            conv_code = f"(1 if {conv_code} else 0)"

        # Now do the actual casting.
        if out_dtype_s == "bool":
            conv_code = f"dace.bool_({conv_code})"
        elif hasattr(dace.dtypes, str(out_dtype)):
            conv_code = f"dace.{out_dtype!s}({conv_code})"
        else:
            raise NotImplementedError(
                f"Cannot convert '{in_dtype}' to '{out_dtype}' as this type is not known to DaCe."
            )

        # Now writing the full Tasklet, i.e. with the output.
        return f"__out0 = {conv_code}"


_ = translator.register_primitive_translator(ConvertElementTypeTranslator())