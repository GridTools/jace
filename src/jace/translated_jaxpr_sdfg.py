# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended versions of `SDFG` and `CompiledSDFG` with additional metadata."""

from __future__ import annotations

import dataclasses
import pathlib
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import dace
from dace import data as dace_data

from jace import util


if TYPE_CHECKING:
    import jax
    from dace.codegen import compiled_sdfg as dace_csdfg


@dataclasses.dataclass(frozen=True, kw_only=True)
class TranslatedJaxprSDFG:
    """
    Encapsulates the SDFG generated from a Jaxpr and additional metadata.

    Contrary to the SDFG that is encapsulated inside an `TranslationContext`
    object, `self` carries a proper SDFG with the following structure:
    - It does not have `__return*` variables, instead all return arguments are
        passed by arguments.
    - All input arguments are passed through arguments mentioned in `input_names`,
        while the outputs are passed through `output_names`.
    - Only variables listed as in/outputs are non transient.
    - The order of `input_names` and `output_names` is the same as in the Jaxpr.
    - Its `arg_names` is set to `input_names + output_names`, but arguments that are
        input and outputs are only listed as inputs.
    - For every transient there is exactly one access node that writes to it,
        except the name of the array starts with `__jace_mutable_`, which can
        be written to multiple times.

    The only valid way to obtain a `TranslatedJaxprSDFG` is by passing a
    `TranslationContext`, that was in turn constructed by
    `JaxprTranslationBuilder.translate_jaxpr()`, to the
    `finalize_translation_context()` or preferably to the `postprocess_jaxpr_sdfg()`
    function.

    Attributes:
        sdfg: The encapsulated SDFG object.
        input_names: SDFG variables used as inputs.
        output_names: SDFG variables used as outputs.

    Todo:
        After the SDFG is compiled a lot of code looks strange, because there is
        no container to store the compiled SDFG and the metadata. This class
        should be extended to address this need.
    """

    sdfg: dace.SDFG
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def validate(self) -> bool:
        """Validate the underlying SDFG."""
        if any(self.sdfg.arrays[inp].transient for inp in self.input_names):
            raise dace.sdfg.InvalidSDFGError(
                f"Found transient inputs: {(inp for inp in self.input_names if self.sdfg.arrays[inp].transient)}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if any(self.sdfg.arrays[out].transient for out in self.output_names):
            raise dace.sdfg.InvalidSDFGError(
                f"Found transient outputs: {(out for out in self.output_names if self.sdfg.arrays[out].transient)}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if self.sdfg.free_symbols:  # This is a simplification that makes our life simple.
            raise dace.sdfg.InvalidSDFGError(
                f"Found free symbols: {self.sdfg.free_symbols}",
                self.sdfg,
                self.sdfg.node_id(self.sdfg.start_state),
            )
        if (self.output_names is not None and self.input_names is not None) and (
            set(self.output_names).intersection(self.input_names)
        ):
            raise dace.sdfg.InvalidSDFGError(
                f"Inputs can not be outputs: {set(self.output_names).intersection(self.input_names)}.",
                self.sdfg,
                None,
            )
        self.sdfg.validate()
        return True


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompiledJaxprSDFG:
    """
    Compiled version of a `TranslatedJaxprSDFG` instance.

    Essentially this class is a wrapper around DaCe's `CompiledSDFG` object, that
    supports the calling convention used inside JaCe, as in `DaCe` it is callable.
    The only valid way to obtain a `CompiledJaxprSDFG` instance is through
    `compile_jaxpr_sdfg()`.

    Args:
        compiled_sdfg: The `CompiledSDFG` object.
        input_names: Names of the SDFG variables used as inputs.
        output_names: Names of the SDFG variables used as outputs.

    Attributes:
        compiled_sdfg: The `CompiledSDFG` object.
        sdfg: SDFG object used to generate/compile `self.compiled_sdfg`.
        input_names: Names of the SDFG variables used as inputs.
        output_names: Names of the SDFG variables used as outputs.

    Note:
        Currently the strides of the input arguments must match the ones that were used
        for lowering the SDFG.
        In DaCe the return values are allocated on a per `CompiledSDFG` basis. Thus
        every call to a compiled SDFG will override the value of the last call, in JaCe
        the memory is allocated on every call. In addition scalars are returned as
        arrays of length one.
    """

    compiled_sdfg: dace_csdfg.CompiledSDFG
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    @property
    def sdfg(self) -> dace.SDFG:  # noqa: D102 [undocumented-public-method]
        return self.compiled_sdfg.sdfg

    def __call__(
        self,
        flat_call_args: Sequence[Any],
    ) -> list[jax.Array]:
        """
        Run the compiled SDFG using the flattened input.

        The function will not perform flattening of its input nor unflattening of
        the output.

        Args:
            flat_call_args: Flattened input arguments.
        """
        if len(self.input_names) != len(flat_call_args):
            raise RuntimeError(
                f"Expected {len(self.input_names)} flattened arguments, but got {len(flat_call_args)}."
            )

        sdfg_call_args: dict[str, Any] = {}
        for in_name, in_val in zip(self.input_names, flat_call_args):
            # TODO(phimuell): Implement a stride matching process.
            if util.is_jax_array(in_val):
                if not util.is_fully_addressable(in_val):
                    raise ValueError(f"Passed a not fully addressable JAX array as '{in_name}'")
                in_val = in_val.__array__()  # noqa: PLW2901 [redefined-loop-name]  # JAX arrays do not expose the __array_interface__.
            sdfg_call_args[in_name] = in_val

        # Allocate the output arrays.
        #  In DaCe the output arrays are created by the `CompiledSDFG` calls and all
        #  calls share the same arrays. In JaCe the output arrays are distinct.
        arrays = self.sdfg.arrays
        for output_name in self.output_names:
            sdfg_call_args[output_name] = dace_data.make_array_from_descriptor(arrays[output_name])

        assert len(sdfg_call_args) == len(self.compiled_sdfg.argnames), (
            "Failed to construct the call arguments,"
            f" expected {len(self.compiled_sdfg.argnames)} but got {len(flat_call_args)}."
            f"\nExpected: {self.compiled_sdfg.argnames}\nGot: {list(sdfg_call_args.keys())}"
        )

        # Calling the SDFG
        with dace.config.temporary_config():
            dace.Config.set("compiler", "allow_view_arguments", value=True)
            self.compiled_sdfg(**sdfg_call_args)

        # DaCe writes the results either into CuPy or NumPy arrays. For compatibility
        #  with JAX we will now turn them into `jax.Array`s. Note that this is safe
        #  because we created these arrays in this function explicitly. Thus when
        #  this function ends, there is no writable reference to these arrays left.
        return [
            util.move_into_jax_array(sdfg_call_args[output_name])
            for output_name in self.output_names
        ]


def compile_jaxpr_sdfg(tsdfg: TranslatedJaxprSDFG) -> dace_csdfg.CompiledJaxprSDFG:
    """Compile `tsdfg` and return a `CompiledJaxprSDFG` object with the result."""
    if any(  # We do not support the DaCe return mechanism
        array_name.startswith("__return") for array_name in tsdfg.sdfg.arrays
    ):
        raise ValueError("Only support SDFGs without '__return' members.")
    if tsdfg.sdfg.free_symbols:  # This is a simplification that makes our life simple.
        raise NotImplementedError(f"No free symbols allowed, found: {tsdfg.sdfg.free_symbols}")
    if not (tsdfg.output_names or tsdfg.input_names):
        raise ValueError("No input nor output.")

    # To ensure that the SDFG is compiled and to get rid of a warning we must modify
    #  some settings of the SDFG. But we also have to fake an immutable SDFG
    sdfg = tsdfg.sdfg
    original_sdfg_name = sdfg.name
    original_recompile = sdfg._recompile
    original_regenerate_code = sdfg._regenerate_code

    try:
        # We need to give the SDFG another name, this is needed to prevent a DaCe
        #  error/warning. This happens if we compile the same lowered SDFG multiple
        #  times with different options.
        sdfg.name = f"{sdfg.name}__{str(uuid.uuid1()).replace('-', '_')}"
        assert len(sdfg.name) < 255  # noqa: PLR2004 [magic-value-comparison]  # 255 maximal file name size on UNIX.

        with dace.config.temporary_config():
            dace.Config.set("compiler", "use_cache", value=False)
            # TODO(egparedes/phimuell): Add a configuration option.
            dace.Config.set("cache", value="name")
            dace.Config.set("default_build_folder", value=pathlib.Path(".jacecache").resolve())
            sdfg._recompile = True
            sdfg._regenerate_code = True
            compiled_sdfg: dace_csdfg.CompiledSDFG = sdfg.compile()

    finally:
        sdfg.name = original_sdfg_name
        sdfg._recompile = original_recompile
        sdfg._regenerate_code = original_regenerate_code

    return CompiledJaxprSDFG(
        compiled_sdfg=compiled_sdfg, input_names=tsdfg.input_names, output_names=tsdfg.output_names
    )
