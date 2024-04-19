# JaCe - JAX jit using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package metadata: version, authors, license and copyright."""

from __future__ import annotations

from typing import Final

from packaging import version as pkg_version


__all__ = ["__author__", "__copyright__", "__license__", "__version__", "__version_info__"]

__author__: Final = "ETH Zurich and individual contributors"
__copyright__: Final = "Copyright (c) 2024 ETH Zurich"
__license__: Final = "BSD-3-Clause-License"
__version__: Final = "0.0.1"
__version_info__: Final = pkg_version.parse(__version__)
