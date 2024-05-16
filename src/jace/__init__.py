# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JaCe - JAX Just-In-Time compilation using DaCe."""

from __future__ import annotations

from .__about__ import __author__, __copyright__, __license__, __version__, __version_info__


def _ensure_build_in_translators_are_loaded() -> None:
    # There is a chicken-egg problem, i.e. circular import, if we use the decorator to add the build in classes.
    #  In order for the decorator to add the translators to the internal list, they have to be run, i.e. imported.
    #  However, since they have to import the decorator, this would lead to a circular import.
    #  To ensure that the built in translators are imported at the beginning, i.e. once Jace is loaded.
    #  We define this function and call it and its only job is to load the subtranslaotrs.
    #  However, this requires that all are imported by the `__init__.py` file.
    # Too see that it is needed, remove this function and run `pytest tests/test_subtranslator_helper.py::test_are_subtranslators_imported`
    from jace.translator import primitive_translators  # noqa: F401  # Unused import


_ensure_build_in_translators_are_loaded()
del _ensure_build_in_translators_are_loaded


__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__version_info__",
]
