# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.metadata
import pathlib
import tempfile

import jace as m


def test_version():
    assert importlib.metadata.version("jace") == m.__version__


def test_folder():
    jacefolder = pathlib.Path(".jacecache/")
    jacefolder.mkdir()


def test_temp_folder():
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)
