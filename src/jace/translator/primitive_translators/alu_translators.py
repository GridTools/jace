# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing all translators related to arithmetic logical operations."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableSequence, Sequence
from typing import Final, cast

import dace
import numpy as np
from jax import core as jax_core
from typing_extensions import override

from jace import translator


class ALUBaseTranslator(translator.PrimitiveTranslator):
    """Base for all ALU (arithmetic logical operations) translators.

    The ALU translators make use of the template pattern and this is the main Skeleton.
    You can think of it as a glorified wrapper around `sdfg::add_mapped_tasklet()`.

    It assumes that Tasklets can be written in the following form:
    ```
        __out0 = f(__in0, __in1, ...)
    ```
    where `f` is some function like plus, i.e. `__out0 = __in0 + __in1`, where `__in{}` is an input connector name of the Tasklet.

    An instance of this class is constructed with the name of the primitive that it should handle and a template.
    The template is basically the code that should be inside the Tasklet, i.e. the function `f`.

    A subclass has to implement the `make_input_memlets()` function which computes the Memlets used as inputs that are used.
    There are two subclasses:
    - `UnaryALUTranslator` for all unary operations.
    - `BinaryALUTranslator` for all binary operations.
    """

    __slots__ = ("_prim_name", "_tskl_tmpl")

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
        self._prim_name = prim_name
        self._tskl_tmpl = tskl_tmpl

    @property
    def primitive(self) -> str:
        """Returns the primitive that should be translated."""
        return self._prim_name

    @override
    def __call__(
        self,
        driver: translator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        """Perform the translation.

        Deepening on the shapes of the input the function will either create a Tasklet or a mapped Tasklet.
        The translator is able to handle broadcasting with NumPy rules.
        The function will always perform the translation inside the provided state.

        Args:
            driver:         The driver object of the translation.
            in_var_names:   List of the names of the arrays created inside the SDFG for the inputs or 'None' in case of a literal.
            out_var_names:  List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The Jax equation that is translated.
            eqn_state:      State into which the primitive's SDFG representation is constructed.
        """
        if len(out_var_names) != 1:
            raise NotImplementedError("'{type(self).__name__}' only one output is allowed.")

        if eqn.outvars[0].aval.shape != ():
            tskl_ranges: list[tuple[str, str]] = [
                (f"__i{dim}", f"0:{N}") for dim, N in enumerate(eqn.outvars[0].aval.shape)
            ]
            tskl_output: dict[str, dace.Memlet] = {
                "__out0": dace.Memlet.simple(
                    out_var_names[0],
                    ", ".join(name for name, _ in tskl_ranges),
                ),
            }
        else:
            # If we have a scalar we will generate a Map, but it will be trivial.
            tskl_ranges = [("__jace_iterator_SCALAR", "0:1")]
            tskl_output = {"__out0": dace.Memlet.simple(out_var_names[0], "0")}

        # Non size dependent properties
        tskl_name: str = f"{self.primitive}_{out_var_names[0]}"
        tskl_code: str = self.write_tasklet_code(in_var_names, eqn)
        tskl_inputs: dict[str, dace.Memlet] = self.make_input_memlets(
            tskl_ranges, in_var_names, eqn
        )

        eqn_state.add_mapped_tasklet(
            name=tskl_name,
            map_ranges=tskl_ranges,
            inputs=tskl_inputs,
            code=tskl_code,
            outputs=tskl_output,
            external_edges=True,
        )

        return eqn_state

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

    @abstractmethod
    def make_input_memlets(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> dict[str, dace.Memlet]:
        """Generate the input Memlets for the non literal operators of the primitive.

        The returned `dict` maps the input connector of the Tasklet to the Memlet that is used to connect it to the Map entry node.

        Args:
            tskl_ranges:    List of the different map parameter, first element is the name of the dimension,
                                    second is the range, i.e. `0:SIZE`.
            in_var_names:       The list of SDFG variables used as input.
            eqn:                The equation object.
        """
        ...


class UnaryALUTranslator(ALUBaseTranslator):
    """Class for all unary operations.

    Thus all Tasklets this class generates have the form:
    ```python
        __out0 = f(__in0)
    ```

    Todo:
        - Specialize for `integer_pow` to do code unrolling in certain situations.
    """

    def make_input_memlets(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> dict[str, dace.Memlet]:
        """Generate the input Memlets for non literal data.

        Args:
            tskl_ranges:    List of the different map parameter, first element is the name of the dimension,
                                second is the range, i.e. `0:SIZE`.
            in_var_names:   The list of SDFG variables used as input.
            eqn:            The equation object.
        """
        in_var_name = in_var_names[0]
        if in_var_name is None:  # Unary operation with literal input -> there is nothing to do.
            return {}
        if eqn.outvars[0].aval.shape == ():
            imemlet = dace.Memlet.simple(in_var_name, "0")
        else:
            imemlet = dace.Memlet.simple(in_var_name, ", ".join(name for name, _ in tskl_ranges))
        return {"__in0": imemlet}


class BinaryALUTranslator(ALUBaseTranslator):
    """Class for all binary ALU operations.

    Thus all Tasklets will have the following form:
    ```python
        __out0 = f(__in0, __in1)
    ```

    The main difference towards the `UnaryALUTranslator` is that this class supports broadcasting.
    However, this is only possible if both operators have the same rank.

    Notes:
        The input `__in0` is identified with the left hand side of an operator and `__in1` is identified as the right hand side.
    """

    def make_input_memlets(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> dict[str, dace.Memlet]:
        out_shps = tuple(eqn.outvars[0].aval.shape)  # Shape of the output
        inp_shpl = tuple(eqn.invars[0].aval.shape)  # Shape of the left/first input
        inp_shpr = tuple(eqn.invars[1].aval.shape)  # Shape of the right/second input

        # Which dimensions on which input should be broadcast, i.e. replicated.
        #  A dimension that is replicated is always accessed with the index `0` in the Memlet.
        #  If `dims_to_bcast*` is `None` then the corresponding argument is a scalar.
        dims_to_bcastl: list[int] | None = []
        dims_to_bcastr: list[int] | None = []

        if out_shps == ():
            # Output is scalar (thus also the inputs).
            dims_to_bcastl = None
            dims_to_bcastr = None

        elif inp_shpl == inp_shpr:
            # The two have the same shapes and neither is a scalar.
            pass

        elif inp_shpl == ():
            # The LHS is a scalar (RHS is not)
            dims_to_bcastl = None

        elif inp_shpr == ():
            # The RHS is a scalar (LHS is not)
            dims_to_bcastr = None

        else:
            # This is the general broadcasting case
            #  We assume that both inputs and the output have the same rank, Jax seems to ensure this.
            assert len(out_shps) == len(inp_shpl) == len(inp_shpr)
            for dim, shp_lft, shp_rgt in zip(range(len(out_shps)), inp_shpl, inp_shpr):
                if shp_lft == shp_rgt:
                    pass  # Needed for cases such as `(10, 1, 3) + (10, 1, 1)`.
                elif shp_lft == 1:
                    dims_to_bcastl.append(dim)  # type: ignore[union-attr]  # guaranteed to be not `None`
                else:
                    dims_to_bcastr.append(dim)  # type: ignore[union-attr]

        # Now we will generate the input Memlets.
        tskl_inputs: dict[str, dace.Memlet] = {}
        for i, in_var_name, dims_to_bcast in zip(
            range(2), in_var_names, [dims_to_bcastl, dims_to_bcastr]
        ):
            if in_var_name is None:  # Input is a literal: No Memlet needed
                continue

            if dims_to_bcast is None:
                imemelt = dace.Memlet.simple(in_var_name, "0")  # Scalar
            else:
                imemelt = dace.Memlet.simple(
                    in_var_name,
                    ", ".join(
                        ("0" if i in dims_to_bcast else it_var)
                        for i, (it_var, _) in enumerate(tskl_ranges)
                    ),
                )
            tskl_inputs[f"__in{i}"] = imemelt

        return tskl_inputs


# Contains all the templates for ALU operations.
_ALU_UN_OPS_TMPL: Final[dict[str, str]] = {
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
}
_ALU_BI_OPS_TMPL: Final[dict[str, str]] = {
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
for pname, ptmpl in _ALU_UN_OPS_TMPL.items():
    translator.register_primitive_translator(UnaryALUTranslator(pname, ptmpl))
for pname, ptmpl in _ALU_BI_OPS_TMPL.items():
    translator.register_primitive_translator(BinaryALUTranslator(pname, ptmpl))
