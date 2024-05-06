# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the `jace.jax.stages.Wrapped` protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jace import jax as jjax, translator
from jace.jax import stages


class JaceWrapped(stages.Wrapped):
    """Result class of all jited functions in Jace.

    It is essentially a wrapper around an already jited, i.e. passed to a Jax primitive function.
    The function is then able to compile it if needed.
    However, the wrapped object is itself again tracable, thus it does not break anything.

    Todo:
        Handles pytrees.
        Configuration of the driver?
        Copy the `jax._src.pjit.make_jit()` functionality to remove `jax.make_jaxpr()`.
    """

    __slots__ = ("fun_",)

    _fun: Callable

    def __init__(
        self,
        fun: Callable,
    ) -> None:
        """Creates a wrapped jace jitable object of `jax_prim`."""
        assert fun is not None
        self._fun: Callable = fun

    def lower(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> jjax.Lowered:
        """Lower this function explicitly for the given arguments.

        Performs the first two steps of the AOT steps described above,
        i.e. transformation into Jaxpr and then to SDFG.
        The result is encapsulated into a `Lowered` object.
        """
        import jax as jax_jax

        if len(kwargs) != 0:
            raise NotImplementedError("Currently only positional arguments are supported.")
        # TODO(phimuell): Handle pytrees.
        real_args: tuple[Any, ...] = args
        jaxpr = jax_jax.make_jaxpr(self._fun)(*real_args)
        driver = translator.JaxprTranslationDriver()
        translated_sdfg: translator.TranslatedJaxprSDFG = driver.translate_jaxpr(jaxpr)
        return jjax.JaceLowered(translated_sdfg)

    @property
    def __wrapped__(self) -> Any:
        """Returns the wrapped object."""
        return self._fun
