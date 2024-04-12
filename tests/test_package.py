from __future__ import annotations

import importlib.metadata

import jace as m


def test_version():
    assert importlib.metadata.version("jace") == m.__version__
