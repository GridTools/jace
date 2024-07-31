#! /usr/bin/env python3

# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "packaging>=24.0",
#   "tomlkit>=0.12.4",
#   "typer-slim>=0.12.3",
#   "yamlpath>=3.8.2"
# ]
# ///

"""Script to synchronize requirements across tools."""

from __future__ import annotations

import copy
import pathlib
import types
from collections.abc import Iterable, Mapping
from typing import TypeAlias

import tomlkit
import typer
import yamlpath
from packaging import requirements as pkg_reqs


def load_from_requirements(filename: str) -> list[pkg_reqs.Requirement]:
    requirements = []
    with pathlib.Path(filename).open(encoding="locale") as f:
        for raw_line in f:
            if (end := raw_line.find("#")) != -1:
                raw_line = raw_line[:end]  # noqa: PLW2901 [redefined-loop-name]
            line = raw_line.strip()
            if line and not line.startswith("-"):
                requirements.append(pkg_reqs.Requirement(line))

    return requirements


def load_from_toml(filename: str, key: str) -> list[pkg_reqs.Requirement]:
    with pathlib.Path(filename).open(encoding="locale") as f:
        toml_data = tomlkit.loads(f.read())

    section = toml_data
    for part in key.split("."):
        section = section[part]

    return [pkg_reqs.Requirement(req) for req in section]


def package_id(req: pkg_reqs.Requirement) -> str:
    req = copy.copy(req)
    req.specifier = pkg_reqs.SpecifierSet()
    req.marker = None
    return str(req)


def version(req: pkg_reqs.Requirement, *, pos: int = 0) -> str:
    return list(req.specifier)[pos].version


def make_versions_map(
    requirements: Iterable[pkg_reqs.Requirement],
) -> dict[str, pkg_reqs.Requirement]:
    result = {}
    for r in requirements:
        req_set = list(r.specifier)
        assert (
            len(req_set) == 1 and req_set[0].operator == "=="
        ), f"Expected exact requirement, got: {req_set}"
        result[package_id(r)] = r
    return result


def dump(
    requirements: pkg_reqs.Requirement | Iterable[pkg_reqs.Requirement],
    *,
    template: str | None = None,
) -> str | list[str]:
    template = template or "{req!s}"
    return (
        [template.format(req=req) for req in requirements]
        if isinstance(requirements, Iterable)
        else template.format(req=requirements)
    )


def dump_to_requirements(
    requirements: Iterable[pkg_reqs.Requirement],
    filename: str,
    *,
    template: str | None = None,
    header: str | None = None,
    footer: str | None = None,
) -> None:
    with pathlib.Path(filename).open("w", encoding="locale") as f:
        if header:
            f.write(f"{header}\n")
        f.write("\n".join(dump(requirements, template=template)))
        if footer:
            f.write(f"{footer}\n")
        f.write("\n")


DumpSpec: TypeAlias = (
    str | Iterable[str] | tuple[pkg_reqs.Requirement | Iterable[pkg_reqs.Requirement], str]
)


def dump_to_yaml(requirements_map: Mapping[str, DumpSpec], filename: str) -> None:
    file_path = pathlib.Path(filename)
    logging_args = types.SimpleNamespace(quiet=False, verbose=False, debug=False)
    console_log = yamlpath.wrappers.ConsolePrinter(logging_args)
    yaml = yamlpath.common.Parsers.get_yaml_editor()
    (yaml_data, doc_loaded) = yamlpath.common.Parsers.get_yaml_data(yaml, console_log, file_path)
    assert doc_loaded
    processor = yamlpath.Processor(console_log, yaml_data)

    for key_path, dump_spec in requirements_map.items():
        if isinstance(dump_spec, tuple):
            value, template = dump_spec
        else:
            assert isinstance(dump_spec, (str, Iterable)), f"Invalid dump spec: {dump_spec}"
            value, template = (dump_spec, None)
        match value:
            case str():
                processor.set_value(yamlpath.YAMLPath(key_path), value)
            case pkg_reqs.Requirement():
                processor.set_value(yamlpath.YAMLPath(key_path), dump(value, template=template))
            case Iterable():
                for _ in processor.delete_nodes(yamlpath.YAMLPath(key_path)):
                    pass
                for i, req in enumerate(value):
                    req_str = req if isinstance(req, str) else dump(req, template=template)
                    item_path = yamlpath.YAMLPath(f"{key_path}[{i}]")
                    processor.set_value(item_path, req_str)

    with file_path.open("w") as f:
        yaml.dump(yaml_data, f)


# -- CLI --
app = typer.Typer()


@app.command()
def pull():
    base = load_from_toml("pyproject.toml", "project.dependencies")
    dump_to_requirements(base, "requirements/base.in")
    cuda12 = load_from_toml("pyproject.toml", "project.optional-dependencies.cuda12")
    dump_to_requirements(cuda12, "requirements/cuda12.in", header="-r base.in")


@app.command()
def push():
    base_names = {package_id(r) for r in load_from_toml("pyproject.toml", "project.dependencies")}
    base_versions_map = make_versions_map([
        r for r in load_from_requirements("requirements/base.txt") if package_id(r) in base_names
    ])
    dev_versions_map = make_versions_map(load_from_requirements("requirements/dev.txt"))
    mypy_dev_versions_map = {k: dev_versions_map[k] for k in ("pytest", "typing-extensions")}

    mypy_req_versions = list((base_versions_map | mypy_dev_versions_map).values())

    dump_to_yaml(
        {
            # ruff
            "repos[.repo%https://github.com/astral-sh/ruff-pre-commit].rev": (
                f"v{version(dev_versions_map['ruff'])}"
            ),
            # mypy
            "repos[.repo%https://github.com/pre-commit/mirrors-mypy].rev": f"v{version(dev_versions_map['mypy'])}",
            "repos[.repo%https://github.com/pre-commit/mirrors-mypy].hooks[.id%mypy].additional_dependencies": (
                mypy_req_versions,
                "{req!s}",
            ),
        },
        ".pre-commit-config.yaml",
    )


if __name__ == "__main__":
    app()
