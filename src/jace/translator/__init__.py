# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subpackage containing all the code related to Jaxpr translation"""

from __future__ import annotations

from .jace_subtranslator_interface import JaCeSubTranslatorInterface
from .jaxpr_translator_driver import JaxprTranslationDriver


__all__ = [
    "JaCeSubTranslatorInterface",
    "JaxprTranslationDriver",
]