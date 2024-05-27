# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reimplementation of the `jax.stages` module.

This module reimplements the public classes of that Jax module.
However, they are a big different, because Jace uses DaCe as backend.

As in Jax Jace has different stages, the terminology is taken from [Jax' AOT-Tutorial](https://jax.readthedocs.io/en/latest/aot.html).
- Stage out:
    In this phase we translate an executable python function into Jaxpr.
- Lower:
    This will transform the Jaxpr into an SDFG equivalent.
    As a implementation note, currently this and the previous step are handled as a single step.
- Compile:
    This will turn the SDFG into an executable object, see `dace.codegen.CompiledSDFG`.
- Execution:
    This is the actual running of the computation.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import dace
import jax as _jax

from jace import optimization, translator, util
from jace.jax import translation_cache as tcache
from jace.optimization import CompilerOptions
from jace.translator import post_translation as ptrans
from jace.util import dace_helper


class JaceWrapped(tcache.CachingStage["JaceLowered"]):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()`.
    Calling it results in jit (just-in-time) lowering, compilation and execution.
    It is also possible to lower the function explicitly by calling `self.lower()`.
    This function can be composed with other Jax transformations.

    Todo:
        - Handle pytrees.
        - Handle all options to `jax.jit`.

    Note:
        The tracing of function will always happen with enabled `x64` mode, which is implicitly
        and temporary activated during tracing. Furthermore, the disable JIT config flag is ignored.
    """

    _fun: Callable
    _primitive_translators: dict[str, translator.PrimitiveTranslator]
    _jit_options: dict[str, Any]

    def __init__(
        self,
        fun: Callable,
        primitive_translators: Mapping[str, translator.PrimitiveTranslator],
        jit_options: Mapping[str, Any],
    ) -> None:
        """Creates a wrapped jitable object of `fun`.

        Args:
            fun:                    The function that is wrapped.
            primitive_translators:  The list of subtranslators that that should be used.
            jit_options:            Options to influence the jit process.
        """
        super().__init__()
        # We have to shallow copy both the translator and the jit options.
        #  This prevents that any modifications affect `self`.
        #  Shallow is enough since the translators themselves are immutable.
        self._primitive_translators = dict(primitive_translators)
        self._jit_options = dict(jit_options)
        self._fun = fun

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Executes the wrapped function, lowering and compiling as needed in one step."""

        # If we are inside a traced context, then we forward the call to the wrapped function.
        #  This ensures that Jace is composable with Jax.
        if util.is_tracing_ongoing(*args, **kwargs):
            return self._fun(*args, **kwargs)

        lowered = self.lower(*args, **kwargs)
        compiled = lowered.compile()
        return compiled(*args, **kwargs)

    @tcache.cached_transition
    def lower(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> JaceLowered:
        """Lower this function explicitly for the given arguments.

        Performs the first two steps of the AOT steps described above, i.e. transformation into
        Jaxpr and then to SDFG. The result is encapsulated into a `Lowered` object.
        """
        if len(kwargs) != 0:
            raise NotImplementedError("Currently only positional arguments are supported.")

        # Currently we do not allow memory order beside `C_CONTIGUOUS`.
        #  This is the best place to check for it.
        if not all((not util.is_array(arg)) or arg.flags["C_CONTIGUOUS"] for arg in args):
            raise NotImplementedError("Currently can not handle strides beside 'C_CONTIGUOUS'.")

        # In Jax `float32` is the main datatype, and they go to great lengths to avoid
        #  some aggressive [type promotion](https://jax.readthedocs.io/en/latest/type_promotion.html).
        #  However, in this case we will have problems when we call the SDFG, for some reasons
        #  `CompiledSDFG` does not work in that case correctly, thus we enable it for the tracing.
        with _jax.experimental.enable_x64():
            driver = translator.JaxprTranslationDriver(
                primitive_translators=self._primitive_translators
            )
            jaxpr = _jax.make_jaxpr(self._fun)(*args)
            tsdfg: translator.TranslatedJaxprSDFG = driver.translate_jaxpr(jaxpr)
        ptrans.postprocess_jaxpr_sdfg(tsdfg=tsdfg, fun=self.wrapped_fun)

        return JaceLowered(tsdfg)

    @property
    def wrapped_fun(self) -> Callable:
        """Returns the wrapped function."""
        return self._fun

    def _make_call_description(
        self,
        *args: Any,
    ) -> tcache.StageTransformationDescription:
        """This function computes the key for the `JaceWrapped.lower()` call.

        Currently it is only able to handle positional argument and does not support static arguments.
        The function will fully abstractify its input arguments.
        This function is used by the cache to generate the key.
        """
        call_args = tuple(tcache._AbstractCallArgument.from_value(x) for x in args)
        return tcache.StageTransformationDescription(stage_id=id(self), call_args=call_args)


class JaceLowered(tcache.CachingStage["JaceCompiled"]):
    """Represents the original computation as an SDFG.

    Although, `JaceWrapped` is composable with Jax transformations `JaceLowered` is not.
    A user should never create such an object.

    Todo:
        - Handle pytrees.
    """

    _translated_sdfg: translator.TranslatedJaxprSDFG

    def __init__(
        self,
        tsdfg: translator.TranslatedJaxprSDFG,
    ) -> None:
        """Initialize the lowered object.

        Args:
            tsdfg:      The lowered SDFG with metadata. Must be finalized.

        Notes:
            The passed `tsdfg` will be managed by `self`.
        """
        if not tsdfg.is_finalized:
            raise ValueError("The translated SDFG must be finalized.")
        super().__init__()
        self._translated_sdfg = tsdfg

    @tcache.cached_transition
    def compile(
        self,
        compiler_options: CompilerOptions | None = None,
    ) -> JaceCompiled:
        """Compile the SDFG.

        Returns an object that encapsulates a compiled SDFG object.
        To influence the various optimizations and compile options of Jace you can use the `compiler_options` argument.
        This is a `dict` which are used as arguments to `jace_optimize()`.

        If nothing is specified `jace.optimization.DEFAULT_OPTIMIZATIONS` will be used.
        Before `compiler_options` is forwarded to `jace_optimize()` it is merged with the default options.

        Note:
            The result of this function is cached.
        """
        # We **must** deepcopy before we do any optimization.
        #  The reason is `self` is cached and assumed to be immutable.
        #  Since all optimizations works in place, we would violate this assumption.
        tsdfg: translator.TranslatedJaxprSDFG = copy.deepcopy(self._translated_sdfg)

        optimization.jace_optimize(tsdfg=tsdfg, **self._make_compiler_options(compiler_options))

        return JaceCompiled(
            csdfg=util.compile_jax_sdfg(tsdfg),
            inp_names=tsdfg.inp_names,
            out_names=tsdfg.out_names,
        )

    def compiler_ir(self, dialect: str | None = None) -> translator.TranslatedJaxprSDFG:
        """Returns the internal SDFG.

        The function returns a `TranslatedJaxprSDFG` object.
        It is important that modifying this object in any ways is considered an error.
        """
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._translated_sdfg
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def as_html(self, filename: str | None = None) -> None:
        """Runs the `view()` method of the underlying SDFG.

        This is a Jace extension.
        """
        self.compiler_ir().sdfg.view(filename=filename, verbose=False)

    def as_sdfg(self) -> dace.SDFG:
        """Returns the encapsulated SDFG.

        It is an error to modify the returned object.
        """
        return self.compiler_ir().sdfg

    def _make_call_description(
        self,
        compiler_options: CompilerOptions | None = None,
    ) -> tcache.StageTransformationDescription:
        """This function computes the key for the `self.compile()` call.

        The function only get one argument that is either a `dict` or a `None`, where `None` means `use default argument.
        The function will construct a concrete description of the call using `(name, value)` pairs.
        This function is used by the cache.
        """
        options = self._make_compiler_options(compiler_options)
        call_args = tuple(sorted(options.items(), key=lambda X: X[0]))
        return tcache.StageTransformationDescription(stage_id=id(self), call_args=call_args)

    def _make_compiler_options(
        self,
        compiler_options: CompilerOptions | None,
    ) -> CompilerOptions:
        return optimization.DEFAULT_OPTIMIZATIONS | (compiler_options or {})


class JaceCompiled:
    """Compiled version of the SDFG.

    Todo:
        - Handle pytrees.
    """

    _csdfg: dace_helper.CompiledSDFG  # The compiled SDFG object.
    _inp_names: tuple[str, ...]  # Name of all input arguments.
    _out_names: tuple[str, ...]  # Name of all output arguments.

    def __init__(
        self,
        csdfg: dace_helper.CompiledSDFG,
        inp_names: Sequence[str],
        out_names: Sequence[str],
    ) -> None:
        if (not inp_names) or (not out_names):
            raise ValueError("Input and output can not be empty.")
        self._csdfg = csdfg
        self._inp_names = tuple(inp_names)
        self._out_names = tuple(out_names)

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Calls the embedded computation."""
        return util.run_jax_sdfg(
            self._csdfg,
            self._inp_names,
            self._out_names,
            args,
            kwargs,
        )


#: Known compilation stages in Jace.
Stage = JaceWrapped | JaceLowered | JaceCompiled


__all__ = [
    "Stage",
    "CompilerOptions",  # export for compatibility with Jax.
    "JaceWrapped",
    "JaceLowered",
    "JaceCompiled",
]
