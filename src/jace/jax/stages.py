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
import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import jax as jax_jax
from jax.stages import CompilerOptions

from jace import optimization, translator, util
from jace.jax import translation_cache as tcache
from jace.translator import managing, post_translation as ptrans
from jace.util import dace_helper as jdace


if TYPE_CHECKING:
    pass


class Stage:
    """A distinct step in the compilation chain, see module description for more.

    The concrete steps are implemented in:
    - JaceWrapped
    - JaceLowered
    - JaceCompiled
    """


class JaceWrapped(Stage):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()`.
    Calling it results in jit (just-in-time) lowering, compilation, and execution.
    It can also be explicitly lowered prior to compilation, and the result compiled prior to execution.

    You should not create `JaceWrapped` instances directly, instead you should use `jace.jit`.

    Todo:
        Handles pytrees.
        Copy the `jax._src.pjit.make_jit()` functionality to remove `jax.make_jaxpr()`.
    """

    _fun: Callable
    _sub_translators: Mapping[str, translator.PrimitiveTranslator]
    _jit_ops: Mapping[str, Any]

    # Managed by the caching infrastructure and only defined during `lower()`.
    #  If defined it contains an abstract description of the function arguments.
    _call_description: tcache.CallArgsDescription | None = None

    # Cache for the lowering. Managed by the caching infrastructure.
    _cache: tcache.TranslationCache | None = None

    def __init__(
        self,
        fun: Callable,
        sub_translators: Mapping[str, translator.PrimitiveTranslator],
        jit_ops: Mapping[str, Any],
    ) -> None:
        """Creates a wrapped jace jitable object of `jax_prim`.

        You should not create `JaceWrapped` instances directly, instead you should use `jace.jit`.

        Args:
            fun:                The function that is wrapped.
            sub_translators:    The list of subtranslators that that should be used.
            jit_ops:            All options that we forward to `jax.jit`.

        Notes:
            Both the `sub_translators` and `jit_ops` are shallow copied.
        """
        self._fun: Callable = fun

        # Why do we have to make a copy (shallow copy is enough as the translators themselves are immutable)?
        #  The question is a little bit tricky so let's consider the following situation:
        #  The user has created a Jace annotated function, and calls it, which leads to lowering and translation.
        #  Then he goes on and in the process modifies the internal list of translators.
        #  Then he calls the same annotated function again, then in case the arguments happens to be structurally the same,
        #  lowering and translation will be skipped if the call is still inside the cache, this is what Jax does.
        #  However, if they are different (or a cache eviction has happened), then tracing and translation will happen again.
        #  Thus depending on the situation the user might get different behaviour.
        #  In my expectation, Jace should always do the same thing, i.e. being deterministic, but what?
        #  In my view, the simplest one and the one that is most functional is, to always use the translators,
        #  that were _passed_ (although implicitly) at construction, making it independent on the global state.
        #  One could argue, that the "dynamical modification of the translator list from the outside" is an actual legitimate use case, however, it is not.
        #  Since `JaceWrapped.lower()` is cached, we would have to modify the caching to include the dynamic state of the set.
        #  Furthermore, we would have to implement to make a distinction between this and the normal use case.
        #  Thus we simply forbid it! If this is desired use `jace.jit()` as function to create an object dynamically.
        # We could either here or in `jace.jit` perform the copy, but since `jace.jit` is at the end
        #  just a glorified constructor and "allowing dynamic translator list" is not a use case, see above, we do it here.
        #
        # Because we know that the global state is immutable, we must not copy in this case.
        #  See also `make_call_description()` in the cache implementation.
        if sub_translators is managing._CURRENT_SUBTRANSLATORS_VIEW:
            self._sub_translators = sub_translators
        else:
            # Note this is the canonical way to shallow copy a mapping since `Mapping` does not has `.copy()`
            # and `copy.copy()` can not handle `MappingProxyType`.
            self._sub_translators = dict(sub_translators)

        # Following the same logic as above we should also copy `jit_ops`.
        #  However, do we have to make a shallow copy or a deepcopy?
        #  I looked at the Jax code and it seems that there is nothing that copies it,
        #  so for now we will just go ahead and shallow copy it.
        self._jit_ops = dict(jit_ops)

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
            Add a context manager to disable caching.
        """
        if len(kwargs) != 0:
            raise NotImplementedError("Currently only positional arguments are supported.")

        # TODO(phimuell): Handle pytrees.
        real_args: tuple[Any, ...] = args

        jaxpr = jax_jax.make_jaxpr(self._fun)(*real_args)
        driver = translator.JaxprTranslationDriver(sub_translators=self._sub_translators)
        trans_sdfg: translator.TranslatedJaxprSDFG = driver.translate_jaxpr(jaxpr)
        ptrans.postprocess_jaxpr_sdfg(tsdfg=trans_sdfg, fun=self.wrapped_fun)
        # The `JaceLowered` assumes complete ownership of `trans_sdfg`!
        return JaceLowered(trans_sdfg)

    @property
    def wrapped_fun(self) -> Callable:
        """Returns the wrapped function."""
        return self._fun


class JaceLowered(Stage):
    """Represents the original computation that was lowered to SDFG."""

    # `self` assumes complete ownership of the
    _trans_sdfg: translator.TranslatedJaxprSDFG

    # Cache for the compilation. Managed by the caching infrastructure.
    _cache: tcache.TranslationCache | None = None

    DEF_COMPILER_OPTIONS: Final[dict[str, Any]] = {
        "auto_optimize": True,
        "simplify": True,
    }

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
        self._trans_sdfg = trans_sdfg

    @tcache.cached_translation
    def compile(
        self,
        compiler_options: CompilerOptions | None = None,  # Unused arguments
    ) -> JaceCompiled:
        """Compile the SDFG.

        Returns an Object that encapsulates a compiled SDFG object.
        You can pass a `dict` as argument which are passed to the `jace_optimize()` routine.
        If you pass `None` then the default options are used.
        To disable all optimization, pass an empty `dict`.

        Notes:
            I am pretty sure that `None` in Jax means "use the default option".
                See also `CachedCallDescription.make_call_description()`.
        """
        # We **must** deepcopy before we do any optimization.
        #  There are many reasons for this but here are the most important ones:
        #  All optimization DaCe functions works in place, if we would not copy the SDFG first, then we would have a problem.
        #  Because, these optimization would then have a feedback of the SDFG object which is stored inside `self`.
        #  Thus, if we would run this code `(jaceLoweredObject := jaceWrappedObject.lower()).compile({opti=True})` would return
        #  an optimized object, which is what we intent to do.
        #  However, if we would now call `jaceWrappedObject.lower()` (with the same arguments as before), we should get `jaceLoweredObject`,
        #  since it was cached, but it would actually contain an already optimized SDFG, which is not what we want.
        #  If you think you can remove this line then do it and run `tests/test_decorator.py::test_decorator_sharing`.
        fsdfg: translator.TranslatedJaxprSDFG = copy.deepcopy(self._trans_sdfg)
        optimization.jace_optimize(
            fsdfg, **(self.DEF_COMPILER_OPTIONS if compiler_options is None else compiler_options)
        )
        csdfg: jdace.CompiledSDFG = util.compile_jax_sdfg(fsdfg)

        return JaceCompiled(
            csdfg=csdfg,
            inp_names=fsdfg.inp_names,
            out_names=fsdfg.out_names,
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

    def as_text(self, dialect: str | None = None) -> str:
        """Textual representation of the SDFG.

        By default, the function will return the Json representation of the SDFG.
        However, by specifying `'html'` as `dialect` the function will call `view()` on the underlying SDFG.

        Notes:
            You should prefer `self.as_html()` instead of this function.
        """
        if (dialect is None) or (dialect.upper() == "JSON"):
            return json.dumps(self.compiler_ir().sdfg.to_json())
        if dialect.upper() == "HTML":
            self.as_html()
            return ""  # For the interface
        raise ValueError(f"Unknown dialect '{dialect}'.")

    def cost_analysis(self) -> Any | None:
        """A summary of execution cost estimates.

        Not implemented use the DaCe [instrumentation API](https://spcldace.readthedocs.io/en/latest/optimization/profiling.html) directly.
        """
        raise NotImplementedError()


class JaceCompiled(Stage):
    """Compiled version of the SDFG.

    Contains all the information to run the associated computation.

    Todo:
        Handle pytrees.
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


__all__ = [
    "Stage",
    "CompilerOptions",
    "JaceWrapped",
    "JaceLowered",
    "JaceCompiled",
]
