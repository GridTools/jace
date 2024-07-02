# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Subpackage containing all the code related to the Jaxpr to SDFG translation.

The concrete primitive translators that ships with JaCe are inside the
`primitive_translators` subpackage.
"""

from __future__ import annotations

from .jaxpr_translator_builder import JaxprTranslationBuilder, TranslationContext
from .primitive_translator import (
    PrimitiveTranslator,
    PrimitiveTranslatorCallable,
    get_registered_primitive_translators,
    make_primitive_translator,
    register_primitive_translator,
)


__all__ = [
    "JaxprTranslationBuilder",
    "PrimitiveTranslator",
    "PrimitiveTranslatorCallable",
    "TranslationContext",
    "get_registered_primitive_translators",
    "make_primitive_translator",
    "register_primitive_translator",
]
