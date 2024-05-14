# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jace.jax.stages.Wrapped` protocol."""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from typing import Any

import jax as jax_jax

from jace import translator, util
from jace.jax import stages
from jace.jax.stages import translation_cache as tcache
from jace.translator import post_translation as ptrans


class JaceWrapped(stages.Stage):
    """A function ready to be specialized, lowered, and compiled.

    This class represents the output of functions such as `jace.jit()`.
    Calling it results in jit (just-in-time) lowering, compilation, and execution.
    It can also be explicitly lowered prior to compilation, and the result compiled prior to execution.

    Notes:
        Reimplementation of `jax.stages.Wrapped` protocol.
        Function wrapped by this class are again tracable by Jax.

    Todo:
        Handles pytrees.
        Configuration of the driver?
        Copy the `jax._src.pjit.make_jit()` functionality to remove `jax.make_jaxpr()`.
    """

    _fun: Callable

    def __init__(
        self,
        fun: Callable,
    ) -> None:
        """Creates a wrapped jace jitable object of `jax_prim`."""
        assert fun is not None
        self._fun: Callable = fun

        # Makes that `self` is a true stand-in for `fun`
        #  This will also add a `__wrapped__` property to `self` which is not part of the interface.
        # TODO(phimuell): modify text to make it clear that it is wrapped, Jax does the same.
        update_wrapper(self, self._fun)

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
        """
        if len(kwargs) != 0:
            raise NotImplementedError("Currently only positional arguments are supported.")

        # TODO(phimuell): Handle pytrees.
        real_args: tuple[Any, ...] = args

        jaxpr = jax_jax.make_jaxpr(self._fun)(*real_args)
        driver = translator.JaxprTranslationDriver()
        trans_sdfg: translator.TranslatedJaxprSDFG = driver.translate_jaxpr(jaxpr)

        fin_sdfg: ptrans.FinalizedJaxprSDFG = ptrans.postprocess_jaxpr_sdfg(
            tsdfg=trans_sdfg, fun=self.__wrapped__
        )

        return stages.JaceLowered(fin_sdfg)
