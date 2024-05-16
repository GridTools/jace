# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jace.jax.stages.Wrapped` protocol."""

from __future__ import annotations

import functools as ft
from collections.abc import Callable, Mapping
from typing import Any

import jax as jax_jax

from jace import translator, util
from jace.jax import stages
from jace.jax.stages import translation_cache as tcache
from jace.translator import managing, post_translation as ptrans


class JaceWrapped(stages.Stage):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()`.
    Calling it results in jit (just-in-time) lowering, compilation, and execution.
    It can also be explicitly lowered prior to compilation, and the result compiled prior to execution.

    You should not create `JaceWrapped` instances directly, instead you should use `jace.jit`.

    Notes:
        The wrapped function is accessible through the `__wrapped__` property.

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

        # Makes that `self` is a true stand-in for `fun`
        self._fun: Callable = fun
        ft.update_wrapper(self, self._fun)  # TODO(phimuell): modify text; Jax does the same.

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

        # This allows us to be composable with Jax transformations.
        if util.is_tracing_ongoing(*args, **kwargs):
            return self.__wrapped__(*args, **kwargs)
        # TODO(phimuell): Handle the case of gradients:
        #                   It seems that this one uses special tracers, since they can handle comparisons.
        #                   https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-autodiff
        # TODO(phimuell): Handle the `disable_jit` context manager of Jax.

        # TODO(phimuell): Handle static arguments correctly
        #                   https://jax.readthedocs.io/en/latest/aot.html#lowering-with-static-arguments
        return self.lower(*args, **kwargs).compile()(*args, **kwargs)

    @tcache.cached_translation
    def lower(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> stages.JaceLowered:
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
        ptrans.postprocess_jaxpr_sdfg(tsdfg=trans_sdfg, fun=self.__wrapped__)
        # The `JaceLowered` assumes complete ownership of `trans_sdfg`!
        return stages.JaceLowered(trans_sdfg)
