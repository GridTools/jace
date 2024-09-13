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
from .concatenate_translator import ConcatenateTranslator
from .conditions import condition_translator
from .convert_element_type_translator import ConvertElementTypeTranslator
from .copy_translator import CopyTranslator, DevicePutTranslator
from .gather_translator import GatherTranslator
from .iota_translator import IotaTranslator
from .pjit_translator import PJITTranslator
from .reshape_translator import ReshapeTranslator
from .select_n_translator import SelectNTranslator
from .slicing import SlicingTranslator
from .squeeze_translator import SqueezeTranslator


__all__ = [
    "ArithmeticOperationTranslator",
    "BroadcastInDimTranslator",
    "ConcatenateTranslator",
    "ConvertElementTypeTranslator",
    "CopyTranslator",
    "DevicePutTranslator",
    "GatherTranslator",
    "IotaTranslator",
    "LogicalOperationTranslator",
    "PJITTranslator",
    "ReshapeTranslator",
    "SelectNTranslator",
    "SlicingTranslator",
    "SqueezeTranslator",
    "condition_translator",
]
