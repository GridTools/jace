# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reimplementation of the `jax.stages` module.

The module either imports or reimplements Jax classes.
In case classes/functions are reimplemented they might be slightly different to fit their usage within Jace.

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

import json
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

from jax._src import stages as jax_stages
from jax.stages import CompilerOptions

from jace import translator, util


class Stage(jax_stages.Stage):
    """A distinct step in the compilation chain, see module description for more.

    This class inherent from its Jax counterpart.
    """


class Wrapped(Protocol):
    """A function ready to be specialized, lowered, and compiled.

    This protocol reflects the output of functions such as `jace.jit`.
    Calling it results in jit (just-in-time) lowering, compilation, and execution.
    It can also be explicitly lowered prior to compilation, and the result compiled prior to execution.

    Notes:
        Reimplementation of `jax.stages.Wrapped` protocol.
    """

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Executes the wrapped function, lowering and compiling as needed in one step."""

        # This allows us to be composable with Jax transformations.
        if util.is_tracing_ongoing(*args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

        # TODO(phimuell): Handle static arguments correctly
        #                   https://jax.readthedocs.io/en/latest/aot.html#lowering-with-static-arguments
        return self.lower(*args, **kwargs).optimize().compile()(*args, **kwargs)

    @abstractmethod
    def lower(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Lowered:
        """Lower this function explicitly for the given arguments.

        Performs the first two steps of the AOT steps described above,
        i.e. stage the computation out to Jaxpr and then translate it to SDFG.
        The result is encapsulated into a `Lowered` object.

        Note:
            As a Jace extension this this function might be change such that it just performs
                the staging out of the Jaxpr, i.e. lowering to SDFG might become a separate step.
        """
        ...

    @property
    @abstractmethod
    def __wrapped__(self) -> Callable:
        """Returns the wrapped function.

        This is a Jace extension.
        """
        ...


class Lowered(Stage):
    """A lowered version of a Python function.

    Essentially this object represents an _unoptimized_ SDFG.
    In addition it contains all meta data that is necessary to compile and run it.

    Notes:
        Partial reimplementation of `jax._src.stages.Lowered`.
    """

    def compile(
        self,
        compiler_options: CompilerOptions | None = None,
    ) -> Compiled:
        """Returns a compiled version of the lowered SDFG.

        The SDFG is compiled as-is, i.e. no transformation or optimizations are applied to it.
        For optimization use the `self.optimize()` function to perform _in-place_ optimization.
        """
        raise NotImplementedError

    def optimize(
        self,
        **kwargs: Any,  # noqa: ARG002  # unused arguments
    ) -> Lowered:
        """Perform optimization _inplace_ and return `self`."""
        return self

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

    def as_html(self, filename: str | None = None) -> None:
        """Runs the `view()` method of the underlying SDFG function.

        This is a Jace extension.
        """
        self.compiler_ir().sdfg.view(filename=filename, verbose=False)

    def compiler_ir(self, dialect: str | None = None) -> translator.TranslatedJaxprSDFG:
        """An arbitrary object representation of this lowering.

        The class will return a `TranslatedJaxprSDFG` object.
        Modifying the returned object is undefined behaviour.

        Args:
            dialect: Optional string specifying a lowering dialect (e.g. "SDFG")

        Notes:
            The Jax documentation points out this function is mainly for debugging.
            The Jax version of this function might return `None`, however, in Jace
                it will always succeed.
        """
        raise NotImplementedError()

    def cost_analysis(self) -> Any | None:
        """A summary of execution cost estimates.

        Not implemented use the DaCe [instrumentation API](https://spcldace.readthedocs.io/en/latest/optimization/profiling.html) directly.
        """
        raise NotImplementedError()


class Compiled(Stage):
    """A compiled version of the computation.

    It contains all necessary information to actually run the computation.
    """

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Executes the wrapped computation."""
        raise NotImplementedError
