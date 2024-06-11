# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reimplementation of the `jax.stages` module.

This module reimplements the public classes of that Jax module.
However, they are a bit different, because JaCe uses DaCe as backend.

As in Jax JaCe has different stages, the terminology is taken from
[Jax' AOT-Tutorial](https://jax.readthedocs.io/en/latest/aot.html).
- Stage out:
    In this phase an executable Python function is translated to Jaxpr.
- Lower:
    This will transform the Jaxpr into an SDFG equivalent. As a implementation
    note, currently this and the previous step are handled as a single step.
- Compile:
    This will turn the SDFG into an executable object, see `dace.codegen.CompiledSDFG`.
- Execution:
    This is the actual running of the computation.

As in Jax the `stages` module give access to the last three stages, but not
the first one.
"""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any

from jax import tree_util as jax_tree

from jace import optimization, translator, util
from jace.optimization import CompilerOptions
from jace.translator import pre_post_translation as ptrans
from jace.util import dace_helper, translation_cache as tcache


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import dace

__all__ = [
    "CompilerOptions",  # export for compatibility with Jax.
    "JaCeCompiled",
    "JaCeLowered",
    "JaCeWrapped",
    "Stage",
]

_JACELOWERED_ACTIVE_COMPILE_OPTIONS: CompilerOptions = optimization.DEFAULT_OPTIMIZATIONS.copy()
"""Global set of currently active compilation/optimization options.

These options are used by `JaCeLowered.compile()` to determine which options
are forwarded to the underlying `jace_optimize()` function. It is initialized
to `jace.optimization.DEFAULT_OPTIMIZATIONS` and can be managed through
`update_active_compiler_options()`.
"""


class JaCeWrapped(tcache.CachingStage["JaCeLowered"]):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()` and is
    the first stage in the translation/compilation chain of JaCe. A user should
    never create a `JaCeWrapped` object directly, instead `jace.jit` should be
    used for that. While it supports just-in-time lowering and compilation, by
    just calling it, these steps can also be performed explicitly. The lowering
    performed by this stage is cached, thus if a `JaCeWrapped` object is lowered
    later, with the same argument the result is taken from the cache.
    Furthermore, a `JaCeWrapped` object is composable with all Jax transformations.

    Args:
        fun: The function that is wrapped.
        primitive_translators: The list of primitive translators that that should be used.
        jit_options: Options to influence the jit process.

    Todo:
        - Support keyword arguments and default values of the wrapped function.
        - Support static arguments.

    Note:
        The tracing of function will always happen with enabled `x64` mode,
        which is implicitly and temporary activated while tracing.
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
        assert all(
            param.default is param.empty for param in inspect.signature(fun).parameters.values()
        )
        super().__init__()
        # We have to shallow copy both the translator and the jit options.
        #  This prevents that any modifications affect `self`.
        #  Shallow is enough since the translators themselves are immutable.
        self._primitive_translators = {**primitive_translators}
        # TODO(phimuell): Do we need to deepcopy the options?
        self._jit_options = {**jit_options}
        self._fun = fun

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the wrapped function, lowering and compiling as needed in one step.

        The arguments passed to this function are the same as the wrapped function uses.
        """
        # If we are inside a traced context, then we forward the call to the wrapped function.
        #  This ensures that JaCe is composable with Jax.
        if util.is_tracing_ongoing(*args, **kwargs):
            return self._fun(*args, **kwargs)

        lowered = self.lower(*args, **kwargs)
        compiled = lowered.compile()
        return compiled(*args, **kwargs)

    @tcache.cached_transition
    def lower(self, *args: Any, **kwargs: Any) -> JaCeLowered:
        """Lower this function explicitly for the given arguments.

        Performs the first two steps of the AOT steps described above, i.e.
        trace the wrapped function with the given arguments and stage it out
        to a Jaxpr. Then translate it to SDFG. The result is encapsulated
        inside a `JaCeLowered` object which can later be compiled.

        Note:
            The call to the function is cached. As key an abstract description
            of the call, similar to the tracers used by Jax, is used.
            The tracing is always done with activated `x64` mode.
        """
        if len(kwargs) != 0:
            raise NotImplementedError("Currently only positional arguments are supported.")

        jaxpr, flat_in_vals, outtree = ptrans.trace_and_flatten_function(
            fun=self._fun,
            trace_call_args=args,
            trace_call_kwargs=kwargs,
            trace_options=self._jit_options,
        )
        builder = translator.JaxprTranslationBuilder(
            primitive_translators=self._primitive_translators
        )
        trans_ctx: translator.TranslationContext = builder.translate_jaxpr(jaxpr)
        tsdfg: translator.TranslatedJaxprSDFG = ptrans.postprocess_jaxpr_sdfg(
            trans_ctx=trans_ctx, fun=self.wrapped_fun, call_args=flat_in_vals, outtree=outtree
        )

        # NOTE: `tsdfg` is deepcopied as a side effect of post processing.
        return JaCeLowered(tsdfg)

    @property
    def wrapped_fun(self) -> Callable:
        """Returns the wrapped function."""
        return self._fun

    def _make_call_description(
        self, args_tree: jax_tree.PyTreeDef, flat_args: Sequence[Any]
    ) -> tcache.StageTransformationSpec:
        """This function computes the key for the `JaCeWrapped.lower()` call inside the cache.

        The function will compute a full abstract description on its argument.
        """
        call_args = tuple(tcache._AbstractCallArgument.from_value(x) for x in flat_args)
        return tcache.StageTransformationSpec(
            stage_id=id(self), call_args=tuple(call_args), args_tree=args_tree
        )


class JaCeLowered(tcache.CachingStage["JaCeCompiled"]):
    """Represents the original computation as an SDFG.

    This class is the output type of `JaCeWrapped.lower()` and represents the
    originally wrapped computation as an SDFG. This stage is followed by the
    `JaCeCompiled` stage, by calling `self.compile()`.

    Before the SDFG is optimized the SDFG is optimized, see `JaCeLowered.compile()`
    for more information on this topic.

    Args:
        tsdfg:  The lowered SDFG with metadata. Must be finalized.

    Note:
        `self` will manage the passed `tsdfg` object. Modifying it results in
        undefined behavior. Although `JaCeWrapped` is composable with Jax
        transformations `JaCeLowered` is not. A user should never create such
        an object, instead `JaCeWrapped.lower()` should be used.
        The storage location and stride of an input (in addition to its shape
        and data type) are hard coded into the SDFG. Thus, if a certain stride
        was used for lowering a computation, that stride must also be used
        when the SDFG is called. If the just in time compilation mode is used
        JaCe will take care of this.
    """

    _translated_sdfg: translator.TranslatedJaxprSDFG

    def __init__(self, tsdfg: translator.TranslatedJaxprSDFG) -> None:
        super().__init__()
        self._translated_sdfg = tsdfg

    @tcache.cached_transition
    def compile(self, compiler_options: CompilerOptions | None = None) -> JaCeCompiled:
        """Optimize and compile the lowered SDFG and return a `JaCeCompiled` object.

        This is the transition function of this stage. Before the SDFG is
        compiled, it will be optimized using `jace_optimize()`. The options
        used for this consists of two parts. First there is the (global) set of
        currently active compiler options, which is then merged with the options
        passed through `compiler_options`, which take precedence. Thus
        `compiler_options` describes the delta from the current active set of options.

        See also:
            `get_active_compiler_options()` to inspect the set of currently active
            options and `update_active_compiler_options()` to modify the set.
        """
        # We **must** deepcopy before we do any optimization, because all optimizations are in
        #  place, however, to properly cache stages, stages needs to be immutable.
        tsdfg: translator.TranslatedJaxprSDFG = copy.deepcopy(self._translated_sdfg)
        optimization.jace_optimize(tsdfg=tsdfg, **self._make_compiler_options(compiler_options))

        return JaCeCompiled(
            csdfg=dace_helper.compile_jax_sdfg(tsdfg),
            inp_names=tsdfg.inp_names,
            out_names=tsdfg.out_names,
            outtree=tsdfg.outtree,
        )

    def compiler_ir(self, dialect: str | None = None) -> translator.TranslatedJaxprSDFG:
        """Returns the internal SDFG.

        The function returns a `TranslatedJaxprSDFG` object. Direct modification
        of the returned object is forbidden and will cause undefined behaviour.
        """
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._translated_sdfg
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def view(self, filename: str | None = None) -> None:
        """Runs the `view()` method of the underlying SDFG.

        This will open a browser and display the SDFG.
        """
        self.compiler_ir().sdfg.view(filename=filename, verbose=False)

    def as_sdfg(self) -> dace.SDFG:
        """Returns the encapsulated SDFG.

        Modifying the returned SDFG in any way is undefined behavior.
        """
        return self.compiler_ir().sdfg

    def _make_call_description(
        self, args_tree: jax_tree.PyTreeDef, flat_args: Sequence[Any]
    ) -> tcache.StageTransformationSpec:
        """This function computes the key for the `self.compile()` call inside the cache.

        Contrary to the `JaCeWrapped.lower()` function the call description
        depends on the concrete values of the arguments and takes the global
        compile options into consideration.
        """
        unflatted_args, unflatted_kwargs = jax_tree.tree_unflatten(args_tree, flat_args)
        assert (not len(unflatted_kwargs)) and (len(unflatted_args) == 1)
        options = self._make_compiler_options(unflatted_args[0])

        # The values are stored inside `call_args` and `args_tree` stores the key.
        call_args, args_tree = jax_tree.tree_flatten(options)
        return tcache.StageTransformationSpec(
            stage_id=id(self), call_args=tuple(call_args), args_tree=args_tree
        )

    def _make_compiler_options(self, compiler_options: CompilerOptions | None) -> CompilerOptions:
        """Return the compilation options that should be used for compilation.

        See `JaCeLowered.compile()` to see how to influence them.
        """
        assert isinstance(compiler_options, dict)
        return get_active_compiler_options() | (compiler_options or {})


def update_active_compiler_options(new_active_options: CompilerOptions) -> CompilerOptions:
    """Updates the set of active compiler options.

    Merges the options passed as `new_active_options` with the currently active
    compiler options. This set is used by `JaCeLowered.compile()` to determine
    which options should be used for optimization.
    The function will return the set of options that was active before the call.
    """
    previous_active_options = _JACELOWERED_ACTIVE_COMPILE_OPTIONS.copy()
    _JACELOWERED_ACTIVE_COMPILE_OPTIONS.update(new_active_options)
    return previous_active_options


def get_active_compiler_options() -> CompilerOptions:
    """Returns the set of currently active compiler options.

    By default the set is initialized with `jace.optimization.DEFAULT_OPTIMIZATIONS`.
    """
    return _JACELOWERED_ACTIVE_COMPILE_OPTIONS.copy()


class JaCeCompiled:
    """Compiled version of the SDFG.

    This is the last stage of the jit chain. A user should never create a
    `JaCeCompiled` instance, instead `JaCeLowered.compile()` should be used.

    In order to execute the stored computation properly, an input's stride,
    storage location, shape and datatype has to match the argument that was
    used for lowering, i.e. was passed to the `lower()` function.

    Args:
        csdfg: The compiled SDFG object.
        inp_names: Names of the SDFG variables used as inputs.
        out_names: Names of the SDFG variables used as outputs.
        outtree: A pytree describing how to unflatten the output.

    Note:
        The class assumes ownership of its input arguments.

    Todo:
        - Automatic strides adaption.
    """

    _csdfg: dace_helper.CompiledSDFG
    _inp_names: tuple[str, ...]
    _out_names: tuple[str, ...]
    _outtree: jax_tree.PyTreeDef

    def __init__(
        self,
        csdfg: dace_helper.CompiledSDFG,
        inp_names: Sequence[str],
        out_names: Sequence[str],
        outtree: jax_tree.PyTreeDef,
    ) -> None:
        if not (out_names or inp_names):
            raise ValueError("No input nor output.")
        self._csdfg = csdfg
        self._inp_names = tuple(inp_names)
        self._out_names = tuple(out_names)
        self._outtree = outtree

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the embedded computation.

        The arguments must be the same as for the wrapped function, but with
        all static arguments removed.
        """
        flat_in_vals = jax_tree.tree_leaves((args, kwargs))
        assert len(flat_in_vals) == len(self._inp_names), "Static arguments."
        return dace_helper.run_jax_sdfg(
            self._csdfg, self._inp_names, self._out_names, flat_in_vals, self._outtree
        )


#: Known compilation stages in JaCe.
Stage = JaCeWrapped | JaCeLowered | JaCeCompiled
