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

import dace
from jax import core as jax_core
from typing_extensions import final, override

from jace import translator


class MappedOperationBaseTranslator(translator.PrimitiveTranslator):
    """Implements the base for all "mapped base operations".

    A mapped base operation `f` is an operation that has several inputs arrays that are elementwise combined to a single output array.
    A prime example for this would be the addition of two arrays of the _same_ size.
    Essentially it assumes that the Tasklet code can be written as:
    ```
        __out0 = f(__in0, __in1, __in3, ...)
    ```
    where `__in*` are the connector names of the Tasklet and `__out0` is the output connector.
    For problems such as this, the SDFG API provides the `SDFGState::add_mapped_tasklet()` function, however, in most cases it can not be directly used.
    Thus this class acts like a convenience wrapper around it.

    To use this class a user has to overwrite the `write_tasklet_code()` function.
    This function generates the Python code that should be put inside the Tasklet.

    Notes:
        This class will always generate a mapped Tasklet, even if a scalar is handled.
        The class will always map over the entirety of the output and assume that all inputs have the same shape as the output.
            If you want to override this behaviour you have to override the `make_input_memlets()` method
            and generate the appropriate Memlets to use as inputs yourself.
        Only one output is allowed.
    """

    __slots__ = ("_prim_name",)

    def __init__(
        self,
        primitive_name: str,
    ) -> None:
        """Bind `self` to the primitive with name `primitive_name`."""
        self._prim_name = primitive_name

    @property
    def primitive(self) -> str:
        """Returns the primitive that should be translated."""
        return self._prim_name

    @final
    @override
    def __call__(
        self,
        driver: translator.JaxprTranslationDriver,
        in_var_names: Sequence[str | None],
        out_var_names: MutableSequence[str],
        eqn: jax_core.JaxprEqn,
        eqn_state: dace.SDFGState,
    ) -> None:
        """Create the mapped Tasklet.

        The function will create the map ranges and based on the shape of the output array.
        It will then call `make_input_memlets()` to get the input Memlets.
        After that it calls `write_tasklet_code()` to get the Tasklet code.
        After that it will create the mapped Tasklet.

        Args:
            driver:         The driver object of the translation.
            in_var_names:   List of the names of the arrays created inside the SDFG for the inputs or 'None' in case of a literal.
            out_var_names:  List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The Jax equation that is translated.
            eqn_state:      State into which the primitive's SDFG representation is constructed.
        """
        if eqn.outvars[0].aval.shape != ():
            tskl_ranges: list[tuple[str, str]] = [
                (f"__i{dim}", f"0:{N}") for dim, N in enumerate(eqn.outvars[0].aval.shape)
            ]
            tskl_output: dict[str, dace.Memlet] = {
                "__out0": dace.Memlet.simple(
                    out_var_names[0],
                    ", ".join(name for name, _ in tskl_ranges),
                )
            }

        else:
            # If we have a scalar we will generate a Map, but it will be trivial.
            tskl_ranges = [("__jace_iterator_SCALAR", "0:1")]
            tskl_output = {"__out0": dace.Memlet.simple(out_var_names[0], "0")}

        tskl_inputs: dict[str, dace.Memlet] = self.make_input_memlets(
            tskl_ranges, in_var_names, eqn
        )
        tskl_name: str = f"{self.primitive}_{out_var_names[0]}"
        tskl_code: str = self.write_tasklet_code(in_var_names, eqn)

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
        ...

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
        if any(eqn.outvars[0].aval.shape != invar.aval.shape for invar in eqn.invars):
            # If you want to use this class as base, then you must override this function.
            raise NotImplementedError(
                "`MappedOperationBaseTranslator` can only handle inputs and output of the same shape!\nEqn: {eqn}"
            )

        return {
            f"__in{i}": dace.Memlet.simple(
                in_var_name,
                (
                    ", ".join(name for name, _ in tskl_ranges)
                    if eqn.outvars[0].aval.shape != ()
                    else "0"
                ),
            )
            for i, in_var_name in enumerate(in_var_names)
            if in_var_name is not None
        }
