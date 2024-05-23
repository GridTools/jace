# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subpackage containing all the code related to Jaxpr translation"""

from __future__ import annotations

from .jaxpr_translator_driver import JaxprTranslationDriver
from .managing import (
    get_regsitered_primitive_translators,
    make_primitive_translator,
    register_primitive_translator,
    set_active_primitive_translators_to,
)
from .primitive_translator import PrimitiveTranslator, PrimitiveTranslatorCallable
from .translated_jaxpr_sdfg import TranslatedJaxprSDFG


__all__ = [
    "JaxprTranslationDriver",
    "PrimitiveTranslator",
    "PrimitiveTranslatorCallable",
    "TranslatedJaxprSDFG",
    "register_primitive_translator",
    "get_regsitered_primitive_translators",
    "set_active_primitive_translators_to",
    "make_primitive_translator",
]
