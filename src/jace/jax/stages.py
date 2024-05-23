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
from jace.translator import pre_post_translation as pptrans
from jace.util import dace_helper as jdace


class JaceWrapped(tcache.CachingStage):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()`.
    Calling it results in jit (just-in-time) lowering, compilation, and execution.
    It can also be explicitly lowered prior to compilation, and the result compiled prior to execution.

    You should not create `JaceWrapped` instances directly, instead you should use `jace.jit`.

    Todo:
        - Handle pytrees.
    """

    _fun: Callable
    _sub_translators: dict[str, translator.PrimitiveTranslator]
    _jit_ops: dict[str, Any]

    def __init__(
        self,
        fun: Callable,
        sub_translators: Mapping[str, translator.PrimitiveTranslator],
        jit_ops: Mapping[str, Any],
    ) -> None:
        """Creates a wrapped jitable object of `fun`.

        You should not create `JaceWrapped` instances directly, instead you should use `jace.jit`.

        Args:
            fun:                The function that is wrapped.
            sub_translators:    The list of subtranslators that that should be used.
            jit_ops:            All options that we forward to `jax.jit`.
        """
        super().__init__()
        # We have to shallow copy both the translator and the jit options.
        #  This prevents that any modifications affect `self`.
        #  Shallow is enough since the translators themselves are immutable.
        self._sub_translators = dict(sub_translators)
        self._jit_ops = dict(jit_ops)
        self._fun = fun

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Executes the wrapped function, lowering and compiling as needed in one step."""

        # TODO(phimuell): Handle the `disable_jit` context manager of Jax.

        # This allows us to be composable with Jax transformations.
        if util.is_tracing_ongoing(*args, **kwargs):
            # TODO(phimuell): Handle the case of gradients:
            #                   It seems that this one uses special tracers, since they can handle comparisons.
            #                   https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-autodiff
            return self._fun(*args, **kwargs)

        # TODO(phimuell): Handle static arguments correctly
        #                   https://jax.readthedocs.io/en/latest/aot.html#lowering-with-static-arguments
        lowered = self.lower(*args, **kwargs)
        compiled = lowered.compile()
        return compiled(*args, **kwargs)

    @tcache.cached_translation
    def lower(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> JaceLowered:
        """Lower this function explicitly for the given arguments.

        Performs the first two steps of the AOT steps described above,
        i.e. transformation into Jaxpr and then to SDFG.
        The result is encapsulated into a `Lowered` object.

        Todo:
            - Handle pytrees.
        """
        if len(kwargs) != 0:
            raise NotImplementedError("Currently only positional arguments are supported.")

        # Currently we do not allow memory order beside `C_CONTIGUOUS`.
        #  This is the best place to check for it.
        if not all((not util.is_array(arg)) or arg.flags["C_CONTIGUOUS"] for arg in args):
            raise NotImplementedError("Currently can not handle strides beside 'C_CONTIGUOUS'.")

        jaxpr = _jax.make_jaxpr(self._fun)(*args)
        driver = translator.JaxprTranslationDriver(sub_translators=self._sub_translators)
        trans_sdfg: translator.TranslatedJaxprSDFG = driver.translate_jaxpr(jaxpr)
        pptrans.postprocess_jaxpr_sdfg(tsdfg=trans_sdfg, fun=self.wrapped_fun)
        # The `JaceLowered` assumes complete ownership of `trans_sdfg`!
        return JaceLowered(trans_sdfg)

    @property
    def wrapped_fun(self) -> Callable:
        """Returns the wrapped function."""
        return self._fun

    def _make_call_description(
        self,
        *args: Any,
    ) -> tcache.CachedCallDescription:
        """This function computes the key for the `JaceWrapped.lower()` call.

        Currently it is only able to handle positional argument and does not support static arguments.
        The function will fully abstractify its input arguments.
        This function is used by the cache to generate the key.
        """
        fargs = tuple(tcache._AbstractCallArgument.from_value(x) for x in args)
        return tcache.CachedCallDescription(stage_id=id(self), fargs=fargs)


class JaceLowered(tcache.CachingStage):
    """Represents the original computation that was lowered to SDFG.

    Todo:
        - Handle pytrees.
    """

    # `self` assumes complete ownership of the
    _trans_sdfg: translator.TranslatedJaxprSDFG

    def __init__(
        self,
        trans_sdfg: translator.TranslatedJaxprSDFG,
    ) -> None:
        """Constructs the lowered object."""
        if not trans_sdfg.is_finalized:
            raise ValueError("The translated SDFG must be finalized.")
        if trans_sdfg.inp_names is None:
            raise ValueError("Input names must be defined.")
        if trans_sdfg.out_names is None:
            raise ValueError("Output names must be defined.")
        super().__init__()
        self._trans_sdfg = trans_sdfg

    @tcache.cached_translation
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
        tsdfg: translator.TranslatedJaxprSDFG = copy.deepcopy(self._trans_sdfg)

        # Must be the same as in `_make_call_description()`!
        options = optimization.DEFAULT_OPTIMIZATIONS | (compiler_options or {})
        optimization.jace_optimize(tsdfg=tsdfg, **options)

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
            return self._trans_sdfg
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
    ) -> tcache.CachedCallDescription:
        """This function computes the key for the `self.compile()` call.

        The function only get one argument that is either a `dict` or a `None`, where `None` means `use default argument.
        The function will construct a concrete description of the call using `(name, value)` pairs.
        This function is used by the cache.
        """
        # Must be the same as in `compile()`!
        options = optimization.DEFAULT_OPTIMIZATIONS | (compiler_options or {})
        fargs = tuple(sorted(options.items(), key=lambda X: X[0]))
        return tcache.CachedCallDescription(stage_id=id(self), fargs=fargs)


class JaceCompiled:
    """Compiled version of the SDFG.

    Todo:
        - Handle pytrees.
    """

    _csdfg: jdace.CompiledSDFG  # The compiled SDFG object.
    _inp_names: tuple[str, ...]  # Name of all input arguments.
    _out_names: tuple[str, ...]  # Name of all output arguments.

    def __init__(
        self,
        csdfg: jdace.CompiledSDFG,
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
