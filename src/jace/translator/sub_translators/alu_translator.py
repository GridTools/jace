# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains the `ALUTranslator` which translates all arithmetic and logic primitives."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Final, cast

import dace
import numpy as np
from jax import core as jcore
from typing_extensions import override

from jace import translator as jtranslator
from jace.translator import util as jtutil


class ALUTranslator(jtranslator.JaCeSubTranslatorInterface):
    """This translator handles all arithmetic and logical operations."""

    __slots__ = ("_unary_ops", "_binary_ops")

    # Contains all translation templates for unarry operations.
    _unary_ops: Final[dict[str, str]] = {
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
    # Transformation for all binary operations
    _binary_ops: Final[dict[str, str]] = {
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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the `ALUTranslator`."""
        super().__init__(**kwargs)

    # end def: __init__

    @override
    def get_handled_primitives(self) -> Collection[str] | str:
        """Returns the list of all known primitives."""
        return set(self._unary_ops.keys()).union(self._binary_ops.keys())

    @override
    def can_translate_jaxeqn(
        self,
        driver: jtranslator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jcore.JaxprEqn,
    ) -> bool:
        """Tests if the translator can handle the primitive.

        Notes:
            A user can generally expect that this function returns `True`.
        """
        is_scalar: bool = len(eqn.outvars[0].aval.shape) == 0
        prim_name: str = eqn.primitive.name
        if prim_name in self._unary_ops:
            assert len(eqn.invars) == 1
        elif prim_name in self._binary_ops:
            assert len(eqn.invars) == 2
        elif out_var_names[0] is None or (not is_scalar) and all(x is None for x in in_var_names):
            return False
        if all(x is None for x in in_var_names):
            return False
        if len(eqn.effects) != 0:
            return False
        return True

    @override
    def translate_jaxeqn(
        self,
        driver: jtranslator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jcore.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        """Perform the translation.

        Deepening on the shapes of the input the function will either create a Tasklet or a mapped Tasklet.
        The translator is able to handle broadcasting with NumPy rules.
        The function will always perform the translation inside the provided state.

        Args:
            driver:         The driver object of the translation.
            in_var_names:   List of the names of the arrays created inside the SDFG for the inpts or 'None' in case of a literal.
            out_var_names:  List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The Jax equation that is translated.
            eqn_state:      State into which the primitive's SDFG representation is constructed.
        """

        # All this checks are done to capture corner cases that Jax might do but are not implemented.
        #  If you find out that Jax will never create something, then remove it.
        if not all(
            invar.aval.shape == ()
            for invar, x in zip(eqn.invars, in_var_names, strict=False)
            if x is None
        ):
            raise NotImplementedError("Can not handle Literals that are not scalars.")
        if in_var_names[0] is None:
            raise NotImplementedError(
                "Literal can only not be on the right hand side of the operation."
            )
        if not any(invar.aval.shape == eqn.outvars[0].aval.shape for invar in eqn.invars):
            raise NotImplementedError("At least input must have the same shape as the output.")
        if len(eqn.outvars) != 1:
            raise NotImplementedError(f"Can only handle one output (Eq: '{eqn}'.")

        # Determine what kind of input we got and how we should proceed.
        is_scalar = len(eqn.outvars[0].aval.shape) == 0
        inp_scalars = [len(Inp.aval.shape) == 0 for i, Inp in enumerate(eqn.invars)]
        has_scalars_as_inputs = any(inp_scalars)
        only_scalars_as_inputs = all(inp_scalars)
        has_some_literals = any(x is None for x in in_var_names)
        only_literals_as_inputs = all(x is None for x in in_var_names)
        inps_same_shape = all(
            eqn.invars[0].aval.shape == eqn.invars[i].aval.shape for i in range(1, len(eqn.invars))
        )

        # We will now look which dimensions have to be broadcasted on which operator.
        #  I.e. in the dimensions in the lists below there will be no map iteration index.
        dims_to_bcastl: list[int] = []
        dims_to_bcastr: list[int] = []

        # Determine if and if yes how we have to broadcast.
        if is_scalar:
            # The output is a scalar, in which case we must have only scalar input as well.
            #  Furthermore, we can have the situation that only Literals are the input.
            assert (not is_scalar) or only_scalars_as_inputs
            assert (not is_scalar) or only_literals_as_inputs

        elif only_literals_as_inputs:
            raise NotImplementedError("Only literals an input is only allowed for the scalar case.")

        elif inps_same_shape:
            pass

        elif has_some_literals or has_scalars_as_inputs:
            # This is essentially array plus scalar, but in two possibilities.
            #  We either have a scalar variable or we have a scalar literal.
            assert (not has_some_literals) or all(
                invar.aval.shape == eqn.outvars[0].aval.shape
                for (invar, x) in zip(eqn.invars, in_var_names, strict=False)
                if x is not None
            )
            assert (not has_scalars_as_inputs) or all(
                invar.aval.shape in {eqn.outvars[0].aval.shape, ()}
                for (invar, x) in zip(eqn.invars, in_var_names, strict=False)
                if x is not None
            )

        elif len(in_var_names) != 2:
            raise ValueError("Can only do broadcasting if there are two operands.")

        else:
            # This is the general broadcasting case
            #  We assume that both inputs and the output have the same rank but different sizes in each dimension.
            #  It seems that Jax ensures this.
            #  We further assume that if the size in a dimension differs then one must have size 1.
            #  This is the size we broadcast over, i.e. conceptually replicated.
            out_shp = tuple(eqn.outvars[0].aval.shape)  # Shape of the output.
            inp_shpl = tuple(eqn.invars[0].aval.shape)  # Shape of the left/first input
            inp_shpr = tuple(
                eqn.invars[1].aval.shape
            )  # Shape of the right/second input; this must be "expanded"

            if not ((len(inp_shpl) == len(inp_shpr)) and (len(out_shp) == len(inp_shpr))):
                raise NotImplementedError("Can not broadcast over different ranks.")

            for dim in reversed(range(len(out_shp))):
                shp_lft = inp_shpl[dim]
                shp_rgt = inp_shpr[dim]

                if shp_lft == shp_rgt:
                    assert out_shp[dim] == shp_lft
                elif shp_lft == 1:
                    assert shp_rgt == out_shp[dim]
                    dims_to_bcastl.append(dim)
                elif shp_rgt == 1:
                    assert shp_lft == out_shp[dim]
                    dims_to_bcastr.append(dim)
                else:
                    raise ValueError(f"Invalid shapes in dimension {dim} for broadcasting.")

        # Now we create the Tasklet into which we solve the equation.
        tskl_code: str = self._writeTaskletCode(in_var_names, eqn)
        tskl_name: str = eqn.primitive.name
        tskl_map_ranges: list[tuple[str, str]] = [
            (f"__i{dim}", f"0:{N}") for dim, N in enumerate(eqn.outvars[0].aval.shape)
        ]
        tskl_outputs: list[tuple[str, dace.Memlet]] = []
        tskl_inputs: list[tuple[str, dace.Memlet] | tuple[None, None]] = []

        # Generate the Memlets for the input.
        for i, dims_to_bcast in zip(
            range(len(eqn.invars)), [dims_to_bcastl, dims_to_bcastr], strict=False
        ):
            if in_var_names[i] is None:
                # Literal: No input needed.
                tskl_inputs.append((None, None))
                continue
            if inp_scalars[i]:
                # We have a scalar argument.
                i_memlet = dace.Memlet.from_array(
                    in_var_names[i], driver.get_sdfg().arrays[in_var_names[i]]
                )
            else:
                # We have an array argument.
                inputs_: list[str] = []
                for dim, (map_var, _) in enumerate(tskl_map_ranges):
                    if dim in dims_to_bcast:
                        inputs_.append("0")
                    else:
                        inputs_.append(str(map_var))
                i_memlet = dace.Memlet.simple(in_var_names[i], ", ".join(inputs_))
                del inputs_
            tskl_inputs.append((f"__in{i}", i_memlet))

        # Now generate the Memlets for the outputs
        if is_scalar:
            tskl_outputs.append(
                (
                    f"__out{i}",
                    dace.Memlet.from_array(
                        out_var_names[0], driver.get_sdfg().arrays[out_var_names[i]]
                    ),
                )
            )
        else:
            tskl_outputs.append(
                (
                    f"__out{i}",
                    dace.Memlet.simple(
                        out_var_names[0], ", ".join([X[0] for X in tskl_map_ranges])
                    ),
                )
            )

        if is_scalar:
            tskl_tasklet = eqn_state.add_tasklet(
                tskl_name,
                jtutil.list_to_dict(tskl_inputs).keys(),
                jtutil.list_to_dict(tskl_outputs).keys(),
                tskl_code,
            )
            for in_var, (in_connector, in_memlet) in filter(
                lambda X: X[0] is not None, zip(in_var_names, tskl_inputs, strict=False)
            ):
                eqn_state.add_edge(
                    eqn_state.add_read(in_var),
                    None,
                    tskl_tasklet,
                    in_connector,
                    in_memlet,
                )
            for out_var, (out_connector, out_memlet) in zip(
                out_var_names, tskl_outputs, strict=False
            ):
                eqn_state.add_edge(
                    tskl_tasklet,
                    out_connector,
                    eqn_state.add_write(out_var),
                    None,
                    out_memlet,
                )
        else:
            eqn_state.add_mapped_tasklet(
                name=tskl_name,
                map_ranges=jtutil.list_to_dict(tskl_map_ranges),
                inputs=jtutil.list_to_dict(tskl_inputs),
                code=tskl_code,
                outputs=jtutil.list_to_dict(tskl_outputs),
                external_edges=True,
            )

        return eqn_state

    def _writeTaskletCode(
        self,
        in_var_names: Sequence[str | None],
        eqn: jcore.JaxprEqn,
    ) -> str:
        """This function generates the Tasklet code based on a primitive.

        The function will also perform literal substitution and parameter handling.

        Args:
            in_var_names:   The list of SDFG variables used as input.
        """
        t_name = eqn.primitive.name
        if t_name == "integer_pow":
            # INTEGER POWER
            exponent = int(eqn.params["y"])
            if exponent == 0:
                t_code = f"__out0 = dace.{eqn.outvars[0].aval.dtype!s}(1)"
            elif exponent == 1:
                t_code = "__out0 = __in0"
            elif exponent == 2:
                t_code = "__out0 = __in0 * __in0"
            elif exponent == 3:
                t_code = "__out0 = (__in0 * __in0) * __in0"
            elif exponent == 4:
                t_code = "__tmp0 = __in0 * __in0\n__out0 = __tmp0 * __tmp0"
            elif exponent == 5:
                t_code = "__tmp0 = __in0 * __in0\n__tmp1 = __tmp0 * __tmp0\n__out0 = __tmp1 * __in0"
            else:
                t_code = self._unary_ops[t_name]

        else:
            # GENERAL CASE
            if t_name in self._unary_ops:
                t_code = self._unary_ops[t_name]
            elif t_name in self._binary_ops:
                t_code = self._binary_ops[t_name]

        # Now we handle Literal substitution
        for i, in_var_name in enumerate(in_var_names):
            if in_var_name is not None:
                continue

            jax_in_var: jcore.Literal = cast(jcore.Literal, eqn.invars[i])
            if jax_in_var.aval.shape == ():
                t_val = jax_in_var.val
                if isinstance(t_val, np.ndarray):
                    t_val = jax_in_var.val.max()  # I do not know a better way in that case
                t_code = t_code.replace(f"__in{i}", str(t_val))
            else:
                raise ValueError(
                    f"Can not handle the literal case of shape: {jax_in_var.aval.shape}"
                )

        # Now replace the parameters
        if len(eqn.params) != 0:
            t_code = t_code.format(**eqn.params)

        return t_code
