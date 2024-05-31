# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subpackage containing all the code related to the Jaxpr to SDFG translation.

The concrete primitive translators that ships with JaCe are inside the `primitive_translators`
subpackage.
"""

from __future__ import annotations

from .jaxpr_translator_builder import JaxprTranslationBuilder, TranslationContext
from .primitive_translator import (
    PrimitiveTranslator,
    PrimitiveTranslatorCallable,
    get_regsitered_primitive_translators,
    make_primitive_translator,
    register_primitive_translator,
    set_active_primitive_translators_to,
)
from .translated_jaxpr_sdfg import TranslatedJaxprSDFG


__all__ = [
    "JaxprTranslationBuilder",
    "PrimitiveTranslator",
    "PrimitiveTranslatorCallable",
    "TranslatedJaxprSDFG",
    "TranslationContext",
    "get_regsitered_primitive_translators",
    "make_primitive_translator",
    "register_primitive_translator",
    "set_active_primitive_translators_to",
]
