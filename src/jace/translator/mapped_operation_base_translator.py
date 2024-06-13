# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing all translators related to arithmetic logical operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import dace
from typing_extensions import final, override

from jace import translator, util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import core as jax_core


class MappedOperationTranslatorBase(translator.PrimitiveTranslator):
    """
    Implements the base for all "mapped base operations".

    A mapped base operation `f` is an operation that has several inputs arrays
    that are elementwise combined to a single output array. A prime example for
    this would be the addition of two arrays. Essentially it assumes that the
    Tasklet code can be written as:
    ```
        __out = f(__in0, __in1, __in3, ...)
    ```
    where `__in*` are the connector names of the Tasklet and `__out` is the
    output connector. For problems such as this, the SDFG API provides the
    `SDFGState.add_mapped_tasklet()` function, however, in most cases it can not
    be directly used, for various reasons. Thus this class acts like a
    convenience wrapper around it.

    To use this class a user has to overwrite the `write_tasklet_code()` function.
    This function generates the entire code that should be put into the Tasklet,
    include the assignment to `__out`. If needed the translator will perform
    literal substitution on the returned code and broadcast the inputs to match
    the outputs.

    If needed a subclass can also override the `make_input_memlets()` function
    to generate custom input Memlets, such as adding an offset.

    Args:
        primitive_name:     The name of the primitive `self` should bind to.

    Notes:
        This class will always generate a mapped Tasklet, even if a scalar is handled.
    """

    def __init__(self, primitive_name: str) -> None:
        self._prim_name = primitive_name

    @property
    def primitive(self) -> str:
        """Returns the primitive that should be translated."""
        return self._prim_name

    @final
    @override
    def __call__(
        self,
        builder: translator.JaxprTranslationBuilder,
        in_var_names: Sequence[str | None],
        out_var_names: Sequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        """
        Create the mapped Tasklet.

        The function will create the map ranges and based on the shape of the
        output array. It will then call `make_input_memlets()` to get the input
        Memlets. After that it calls `write_tasklet_code()` to get the Tasklet
        code and perform literal substitution by forwarding it to
        `self.literal_substitution()`. After that it will create the mapped Tasklet.

        Note:
            For a description of the arguments see `PrimitiveTranslatorCallable`.
        """
        assert len(out_var_names) == 1
        if util.get_jax_var_shape(eqn.outvars[0]) != ():
            tskl_ranges: list[tuple[str, str]] = [
                (f"__i{dim}", f"0:{N}")
                for dim, N in enumerate(util.get_jax_var_shape(eqn.outvars[0]))
            ]
            tskl_output: dict[str, dace.Memlet] = {
                "__out": dace.Memlet.simple(
                    out_var_names[0], ", ".join(name for name, _ in tskl_ranges)
                )
            }

        else:
            # If we have a scalar we will generate a Map, but it will be trivial.
            tskl_ranges = [("__jace_iterator_SCALAR", "0:1")]
            tskl_output = {"__out": dace.Memlet.simple(out_var_names[0], "0")}

        tskl_inputs: dict[str, dace.Memlet] = self.make_input_memlets(
            tskl_ranges, in_var_names, eqn
        )
        tskl_name = f"{self.primitive}_{out_var_names[0]}"
        tskl_code = self.write_tasklet_code(tskl_ranges, in_var_names, eqn)
        tskl_code = self.literal_substitution(tskl_code, in_var_names, eqn)

        eqn_state.add_mapped_tasklet(
            name=tskl_name,
            map_ranges=tskl_ranges,
            inputs=tskl_inputs,
            code=tskl_code,
            outputs=tskl_output,
            external_edges=True,
        )

        return eqn_state

    @abstractmethod
    def write_tasklet_code(
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> str:
        """
        Return the (Python) code that should be put inside the Tasklet.

        This also includes the assignment statement, i.e. `__out`.
        However, the base will do literal substitution on the returned object.

        Args:
            tskl_ranges:    List of pairs used as map parameter, first element
                is the name iteration index of the dimension, second is its range.
            in_var_names:   The list of SDFG variables used as input, `None` if literal.
            eqn:            The equation.
        """
        ...

    def make_input_memlets(  # noqa: PLR6301  # Subclasses might need them.
        self,
        tskl_ranges: Sequence[tuple[str, str]],
        in_var_names: Sequence[str | None],
        eqn: jax_core.JaxprEqn,
    ) -> dict[str, dace.Memlet]:
        """
        Generate the input Memlets for the non literal operators of the primitive.

        The returned `dict` maps the input connector of the Tasklet to the Memlet
        that is used to connect it to the Map entry node.

        Args:
            tskl_ranges:    List of pairs used as map parameter, first element
                is the name iteration index of the dimension, second is its range
            in_var_names:   The list of SDFG variables used as input, `None` if literal.
            eqn:            The equation object.
        """
        out_shp = tuple(util.get_jax_var_shape(eqn.outvars[0]))  # Shape of the output
        out_rank = len(out_shp)
        if any(len(util.get_jax_var_shape(invar)) not in {0, out_rank} for invar in eqn.invars):
            raise NotImplementedError(
                f"'MappedOperationTranslatorBase' Inputs must have the same rank as the output! "
                f"Eqn: {eqn} || {tuple(util.get_jax_var_shape(eqn.outvars[0]))}"
            )

        # Now we will generate the input Memlets.
        tskl_inputs: dict[str, dace.Memlet] = {}
        for i, (in_var_name, inp_shp) in enumerate(
            zip(in_var_names, (util.get_jax_var_shape(invar) for invar in eqn.invars))
        ):
            if in_var_name is None:  # Input is a literal: No Memlet needed
                continue

            if inp_shp == ():  # Scalars
                tskl_inputs[f"__in{i}"] = dace.Memlet.simple(in_var_name, "0")  # Scalar
                continue

            # We have to to broadcasting (combine yes and no together)
            dims_to_bcast: Sequence[int] = [dim for dim in range(out_rank) if inp_shp[dim] == 1]
            tskl_inputs[f"__in{i}"] = dace.Memlet.simple(
                in_var_name,
                ", ".join(
                    ("0" if i in dims_to_bcast else it_var)
                    for i, (it_var, _) in enumerate(tskl_ranges)
                ),
            )
        return tskl_inputs

    def literal_substitution(  # noqa: PLR6301  # Subclasses might need it.
        self, tskl_code: str, in_var_names: Sequence[str | None], eqn: jax_core.JaxprEqn
    ) -> str:
        """
        Perform literal substitution on the proto Tasklet code `tskl_code`.

        Args:
            tskl_code:      The proto Tasklet code with literal.
            in_var_names:   The list of SDFG variables used as input.
            eqn:            The equation.

        Note:
            It is allowed but not recommended to override this function.
        """
        for i, in_var_name in enumerate(in_var_names):
            if in_var_name is not None:
                continue
            t_val = util.get_jax_literal_value(eqn.invars[i])
            tskl_code = tskl_code.replace(f"__in{i}", str(t_val))
        return tskl_code
