# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Module collecting all built-in primitive translators."""

from __future__ import annotations

from .arithmetic_logical_translators import (
    ArithmeticOperationTranslator,
    LogicalOperationTranslator,
)
from .broadcast_in_dim_translator import BroadcastInDimTranslator
from .concatenate_translator import concatenate_translator
from .conditions import condition_translator
from .convert_element_type_translator import ConvertElementTypeTranslator
from .copy_translator import copy_translator, device_put_translator
from .gather_translator import GatherTranslator
from .iota_translator import IotaTranslator
from .pjit_translator import pjit_translator
from .reshape_translator import reshape_translator
from .select_n_translator import SelectNTranslator
from .slicing import SlicingTranslator, dynamic_slicing_translator
from .squeeze_translator import SqueezeTranslator


__all__ = [
    "ArithmeticOperationTranslator",
    "BroadcastInDimTranslator",
    "ConvertElementTypeTranslator",
    "GatherTranslator",
    "IotaTranslator",
    "LogicalOperationTranslator",
    "SelectNTranslator",
    "SlicingTranslator",
    "SqueezeTranslator",
    "concatenate_translator",
    "condition_translator",
    "copy_translator",
    "device_put_translator",
    "dynamic_slicing_translator",
    "pjit_translator",
    "reshape_translator",
]
