# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.metadata


project = "JaCe"
copyright = "2024, ETH Zurich"  # noqa: A001 [builtin-variable-shadowing]
author = "ETH Zurich"
version = release = importlib.metadata.version("jace")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store", ".env", ".venv"]

html_theme = "furo"

myst_enable_extensions = ["colon_fence"]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

nitpick_ignore = [("py:class", "_io.StringIO"), ("py:class", "_io.BytesIO")]

always_document_param_types = True
