# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subpackage containing all the code related to Jaxpr translation"""

from __future__ import annotations

from .jaxpr_translator_driver import JaxprTranslationDriver
from .managing import get_regsitered_primitive_translators, register_primitive_translator
from .primitive_translator import PrimitiveTranslator, PrimitiveTranslatorCallable
from .translated_jaxpr_sdfg import TranslatedJaxprSDFG


__all__ = [
    "JaxprTranslationDriver",
    "PrimitiveTranslator",
    "PrimitiveTranslatorCallable",
    "TranslatedJaxprSDFG",
    "register_primitive_translator",
    "get_regsitered_primitive_translators",
]
