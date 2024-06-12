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

import pathlib
import re
import types
from collections.abc import Iterable, Mapping
from typing import NamedTuple, TypeAlias

import tomlkit
import typer
import yamlpath
from packaging import (
    markers as pkg_markers,
    requirements as pkg_requirements,
    specifiers as pkg_specifiers,
)


# -- Classes --
class RequirementSpec(NamedTuple):
    """A parsed requirement specification."""

    package: pkg_requirements.Requirement
    specifiers: pkg_specifiers.SpecifierSet | None = None
    marker: pkg_markers.Marker | None = None

    @classmethod
    def from_text(cls, req_text: str) -> RequirementSpec:
        req_text = req_text.strip()
        assert req_text, "Requirement string cannot be empty"

        m = re.match(r"^([^><=~]*)\s*([^;]*)\s*;?\s*(.*)$", req_text)
        return RequirementSpec(
            pkg_requirements.Requirement(m[1]),
            pkg_specifiers.Specifier(m[2]) if m[2] else None,
            pkg_markers.Marker(m[3]) if m[3] else None,
        )

    def as_text(self) -> str:
        return f"{self.package!s}{(self.specifiers or '')!s}{(self.marker or '')!s}".strip()


class Requirement(NamedTuple):
    """An item in a list of requirements and its parsed specification."""

    text: str
    spec: RequirementSpec

    @classmethod
    def from_text(cls, req_text: str) -> Requirement:
        return Requirement(req_text, RequirementSpec.from_text(req_text))

    @classmethod
    def from_spec(cls, req: RequirementSpec) -> Requirement:
        return Requirement(req.as_text(), req)

    def dump(self, *, template: str | None = None) -> str:
        template = template or "{req.text}"
        return template.format(req=self)


class RequirementDumpSpec(NamedTuple):
    value: Requirement | Iterable[Requirement]
    template: str | None = None


DumpSpec: TypeAlias = (
    RequirementDumpSpec | tuple[Requirement | Iterable[Requirement], str | None] | str
)


# -- Functions --
def make_requirements_map(requirements: Iterable[Requirement]) -> dict[str, Requirement]:
    return {req.spec.package.name: req for req in requirements}


def load_from_requirements(filename: str) -> list[Requirement]:
    requirements = []
    with pathlib.Path(filename).open(encoding="locale") as f:
        for raw_line in f:
            if (end := raw_line.find("#")) != -1:
                raw_line = raw_line[:end]  # noqa: PLW2901 [redefined-loop-name]
            line = raw_line.strip()
            if line and not line.startswith("-"):
                requirements.append(Requirement.from_text(line))

    return requirements


def load_from_toml(filename: str, key: str) -> list[Requirement]:
    with pathlib.Path(filename).open(encoding="locale") as f:
        toml_data = tomlkit.loads(f.read())

    section = toml_data
    for part in key.split("."):
        section = section[part]

    return [Requirement.from_text(req) for req in section]


def dump(requirements: Iterable[Requirement], *, template: str | None = None) -> None:
    return [req.dump(template=template) for req in requirements]


def dump_to_requirements(
    requirements: Iterable[Requirement],
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


def dump_to_yaml(requirements_map: Mapping[str, DumpSpec], filename: str) -> None:
    file_path = pathlib.Path(filename)
    logging_args = types.SimpleNamespace(quiet=False, verbose=False, debug=False)
    console_log = yamlpath.wrappers.ConsolePrinter(logging_args)
    yaml = yamlpath.common.Parsers.get_yaml_editor()
    (yaml_data, doc_loaded) = yamlpath.common.Parsers.get_yaml_data(yaml, console_log, file_path)
    assert doc_loaded
    processor = yamlpath.Processor(console_log, yaml_data)

    for key_path, (value, template) in requirements_map.items():
        match value:
            case str():
                processor.set_value(yamlpath.YAMLPath(key_path), value)
            case Requirement():
                processor.set_value(yamlpath.YAMLPath(key_path), value.dump(template=template))
            case Iterable():
                for _ in processor.delete_nodes(yamlpath.YAMLPath(key_path)):
                    pass
                for i, req in enumerate(dump(value, template=template)):
                    item_path = yamlpath.YAMLPath(f"{key_path}[{i}]")
                    processor.set_value(item_path, req)

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
    base_names = {r.spec.package for r in load_from_toml("pyproject.toml", "project.dependencies")}
    base_versions = [
        r for r in load_from_requirements("requirements/base.txt") if r.spec.package in base_names
    ]
    dev_versions_map = make_requirements_map(load_from_requirements("requirements/dev.txt"))
    mypy_req_versions = sorted(
        base_versions + [dev_versions_map[r] for r in ("pytest", "typing-extensions")],
        key=lambda r: str(r.spec.package),
    )
    dump_to_yaml(
        {
            # ruff
            "repos[.repo%https://github.com/astral-sh/ruff-pre-commit].rev": (
                dev_versions_map["ruff"],
                "v{req.spec.specifiers.version}",
            ),
            # mypy
            "repos[.repo%https://github.com/pre-commit/mirrors-mypy].rev": (
                dev_versions_map["mypy"],
                "v{req.spec.specifiers.version}",
            ),
            "repos[.repo%https://github.com/pre-commit/mirrors-mypy].hooks[.id%mypy].additional_dependencies": (
                mypy_req_versions,
                None,
            ),
        },
        ".pre-commit-config.yaml",
    )


if __name__ == "__main__":
    app()
