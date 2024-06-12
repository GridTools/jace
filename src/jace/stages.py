# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reimplementation of the `jax.stages` module.

This module reimplements the public classes of that Jax module.
However, because JaCe uses DaCe as backend they differ is some small aspects.

As in Jax JaCe has different stages, the terminology is taken from
[Jax' AOT-Tutorial](https://jax.readthedocs.io/en/latest/aot.html).
- Stage out:
    In this phase an executable Python function is translated to a Jaxpr.
- Lower:
    This will transform the Jaxpr into its SDFG equivalent.
- Compile:
    This will turn the SDFG into an executable object.
- Execution:
    This is the actual running of the computation.

As in Jax the in JaCe the user only has access to the last tree stages and
staging out and lowering is handled as a single step.
"""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any, Union

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
to `jace.optimization.DEFAULT_OPTIMIZATIONS` and can be managed through the
`update_active_compiler_options()` function.
"""

#: Known compilation stages in JaCe.
Stage = Union["JaCeWrapped", "JaCeLowered", "JaCeCompiled"]


class JaCeWrapped(tcache.CachingStage["JaCeLowered"]):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()` and is
    the first stage in the translation/compilation chain of JaCe. A user should
    never create a `JaCeWrapped` object directly, instead `jace.jit` should be
    used. While it supports just-in-time lowering and compilation, by just
    calling it, these steps can also be performed explicitly.
    The lowering, performed by this stage is cached, thus if a `JaCeWrapped`
    object is later lowered with the same arguments the result might be taken
    from the cache.

    Furthermore, a `JaCeWrapped` object is composable with all Jax transformations.

    Args:
        fun: The function that is wrapped.
        primitive_translators: The list of primitive translators that should
            be used for the lowering to SDFG.
        jit_options: Options to control the lowering process.

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
        """Executes the wrapped function.

        This function will lower and compile in one go.
        The function accepts the same arguments as the original computation.

        Note:
            This function is also aware if a Jax tracing is going on. In this
            case, it will not lower and compile but forward the call to the
            wrapped Python function.
        """
        if util.is_tracing_ongoing(*args, **kwargs):
            return self._fun(*args, **kwargs)

        lowered = self.lower(*args, **kwargs)
        compiled = lowered.compile()
        # TODO(phimuell): Filter out static arguments
        return compiled(*args, **kwargs)

    @tcache.cached_transition
    def lower(self, *args: Any, **kwargs: Any) -> JaCeLowered:
        """Lower the wrapped computation for the given arguments.

        This function accepts the same arguments as the original computation does.

        Performs the first two steps of the AOT steps described above, i.e.
        trace the wrapped function with the given arguments and stage it out
        to a Jaxpr. Then translate it to an SDFG. The result is encapsulated
        inside a `JaCeLowered` object which can later be compiled.

        It should be noted that the current lowering process will hard code
        the strides and the storage location of the input inside the SDFG.
        Thus if the SDFG is lowered with arrays in C order, calling the compiled
        SDFG with FORTRAN order will result in an error.

        Note:
            The tracing is always done with activated `x64` mode.
        """
        jaxpr, flat_call_args, outtree = ptrans.trace_and_flatten_function(
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
            trans_ctx=trans_ctx, fun=self.wrapped_fun, call_args=flat_call_args, outtree=outtree
        )

        # NOTE: `tsdfg` is deepcopied as a side effect of post processing.
        return JaCeLowered(tsdfg)

    @property
    def wrapped_fun(self) -> Callable:
        """Returns the wrapped function."""
        return self._fun

    def _make_call_description(
        self, intree: jax_tree.PyTreeDef, flat_call_args: Sequence[Any]
    ) -> tcache.StageTransformationSpec:
        """Generates the key used to cache lowering calls.

        For all non static arguments the function will generate an abstract
        description of an argument and for all static arguments the concrete
        value.

        Notes:
            The abstract description also includes storage location, i.e. if
            on CPU or on GPU, and the strides of the arrays.
        """
        # TODO(phimuell): Implement static arguments
        flat_call_args = tuple(tcache._AbstractCallArgument.from_value(x) for x in flat_call_args)
        return tcache.StageTransformationSpec(
            stage_id=id(self), flat_call_args=tuple(flat_call_args), intree=intree
        )


class JaCeLowered(tcache.CachingStage["JaCeCompiled"]):
    """Represents the original computation as an SDFG.

    This class is the output type of `JaCeWrapped.lower()` and represents the
    originally wrapped computation as an SDFG. This stage is followed by the
    `JaCeCompiled` stage, by calling `self.compile()`. A user should never
    directly construct a `JaCeLowered` object directly, instead
    `JaCeWrapped.lower()` should be used.

    Before the SDFG is compiled it is optimized, see `JaCeLowered.compile()` for
    how to control the process.

    Args:
        tsdfg:  The lowered SDFG with metadata.

    Note:
        `self` will manage the passed `tsdfg` object. Modifying it results is
        undefined behavior. Although `JaCeWrapped` is composable with Jax
        transformations `JaCeLowered` is not.
    """

    _translated_sdfg: translator.TranslatedJaxprSDFG

    def __init__(self, tsdfg: translator.TranslatedJaxprSDFG) -> None:
        super().__init__()
        self._translated_sdfg = tsdfg

    @tcache.cached_transition
    def compile(self, compiler_options: CompilerOptions | None = None) -> JaCeCompiled:
        """Optimize and compile the lowered SDFG and return a `JaCeCompiled` object.

        Before the SDFG is compiled, it will be optimized using `jace_optimize()`.
        There are two different sources of these options. The first one is the
        global set of currently active compiler options. The second one is the
        options that are passed to this function, which takes precedence. Thus,
        the `compiler_options` argument of this function describes the difference
        from the currently active global options.

        See also:
            `get_active_compiler_options()` to inspect the set of currently active
            options and `update_active_compiler_options()` to modify them.
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
        of the returned object is forbidden and results in undefined behaviour.
        """
        if (dialect is None) or (dialect.upper() == "SDFG"):
            return self._translated_sdfg
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def view(self, filename: str | None = None) -> None:
        """Runs the `view()` method of the underlying SDFG."""
        self.compiler_ir().sdfg.view(filename=filename, verbose=False)

    def as_sdfg(self) -> dace.SDFG:
        """Returns the encapsulated SDFG.

        Modifying the returned SDFG in any way is undefined behavior.
        """
        return self.compiler_ir().sdfg

    def _make_call_description(
        self, intree: jax_tree.PyTreeDef, flat_call_args: Sequence[Any]
    ) -> tcache.StageTransformationSpec:
        """Creates the key for the `self.compile()` transition function.

        The generated key will not only depend on the arguments that were
        passed to the translation function, i.e. `compile(compiler_options)`,
        in addition it will also take the set of currently active set of
        global options. Furthermore, the key will depend on the concrete values.
        """
        unflatted_args, unflatted_kwargs = jax_tree.tree_unflatten(intree, flat_call_args)
        assert (not len(unflatted_kwargs)) and (len(unflatted_args) == 1)

        options = self._make_compiler_options(unflatted_args[0])
        flat_options, optiontree = jax_tree.tree_flatten(options)
        return tcache.StageTransformationSpec(
            stage_id=id(self), flat_call_args=tuple(flat_options), intree=optiontree
        )

    def _make_compiler_options(self, compiler_options: CompilerOptions | None) -> CompilerOptions:
        """Return the compilation options that should be used for compilation."""
        assert isinstance(compiler_options, dict)
        return get_active_compiler_options() | (compiler_options or {})


def update_active_compiler_options(new_active_options: CompilerOptions) -> CompilerOptions:
    """Updates the set of active compiler options.

    Merges the options passed as `new_active_options` with the currently active
    compiler options. This set is used by `JaCeLowered.compile()` to determine
    which options should be used.
    The function will return the set of options that was active before the call.

    To obtain the set of currently active options use `get_active_compiler_options()`.

    Todo:
        Make a proper context manager.
    """
    previous_active_options = _JACELOWERED_ACTIVE_COMPILE_OPTIONS.copy()
    _JACELOWERED_ACTIVE_COMPILE_OPTIONS.update(new_active_options)
    return previous_active_options


def get_active_compiler_options() -> CompilerOptions:
    """Returns the set of currently active compiler options."""
    return _JACELOWERED_ACTIVE_COMPILE_OPTIONS.copy()


class JaCeCompiled:
    """Compiled version of the SDFG.

    This is the last stage of the JaCe's jit chain. A user should never create a
    `JaCeCompiled` instance, instead `JaCeLowered.compile()` should be used.

    Since the strides and storage location of the arguments, that where used
    to lower the computation are hard coded inside the SDFG, a `JaCeCompiled`
    object can only be called with compatible arguments.

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


        Note:
            Unlike the `lower()` function which takes the same arguments as the
            original computation, to call this function you have to remove all
            static arguments.
            Furthermore, all arguments must have strides and storage locations
            that is compatible with the ones that were used for lowering.
        """
        flat_in_vals = jax_tree.tree_leaves((args, kwargs))
        assert len(flat_in_vals) == len(self._inp_names), "Static arguments."
        flat_output = dace_helper.run_jax_sdfg(
            self._csdfg, self._inp_names, self._out_names, flat_in_vals
        )
        return jax_tree.tree_unflatten(self._outtree, flat_output)
