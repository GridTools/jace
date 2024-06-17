"""Nox session definitions."""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil

import nox


nox.needs_version = ">=2024.3.2"
nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"


ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_DEV_VENV_PATH = ROOT_DIR / ".venv"


def load_from_frozen_requirements(filename: str) -> dict[str, str]:
    requirements = {}
    with pathlib.Path(filename).open(encoding="locale") as f:
        for raw_line in f:
            if (end := raw_line.find("#")) != -1:
                raw_line = raw_line[:end]  # noqa: PLW2901 [redefined-loop-name]
            line = raw_line.strip()
            if line and not line.startswith("-"):
                m = re.match(r"^([^=]*)\s*([^;]*)\s*;?\s*(.*)$", line)
                if m:
                    requirements[m[1]] = m[2]

    return requirements


REQUIREMENTS = load_from_frozen_requirements(ROOT_DIR / "requirements" / "dev.txt")


@nox.session(python="3.10")
def lint(session: nox.Session) -> None:
    """Run the linter (pre-commit)."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e", ".", "-r", "requirements/dev.txt")
    session.run("pytest", *session.posargs)


@nox.session(python=["3.10", "3.11", "3.12"])
def venv(session: nox.Session) -> None:
    """
    Sets up a Python development environment. Use as: `nox -s venv-3.xx -- [req_preset] [dest_path]

    req_preset: The requirements file to use as 'requirements/{req_preset}.txt'.
        Default: 'dev'
    dest_path (optional): The path to the virtualenv to create.
        Default: '.venv-{3.xx}-{req_preset}'

    This session will:
    - Create a python virtualenv for the session
    - Install the `virtualenv` cli tool into this environment
    - Use `virtualenv` to create a project virtual environment
    - Invoke the python interpreter from the created project environment
      to install the project and all it's development dependencies.
    """  # noqa: W505 [doc-line-too-long]
    req_preset = "dev"
    venv_path = None
    virtualenv_args = []
    if session.posargs:
        req_preset, *more_pos_args = session.posargs
        if more_pos_args:
            venv_path, *_ = more_pos_args
    if not venv_path:
        venv_path = f"{DEFAULT_DEV_VENV_PATH}-{session.python}-{req_preset}"
    venv_path = pathlib.Path(venv_path).resolve()

    if not venv_path.exists():
        print(f"Creating virtualenv at '{venv_path}' (options: {virtualenv_args})...")
        session.install("virtualenv")
        session.run("virtualenv", venv_path, silent=True)
    elif venv_path.exists():
        assert (
            venv_path.is_dir() and (venv_path / "bin" / f"python{session.python}").exists
        ), f"'{venv_path}' path already exists but is not a virtualenv with python{session.python}."
        print(f"'{venv_path}' path already exists. Skipping virtualenv creation...")

    python_path = venv_path / "bin" / "python"
    requirements_file = f"requirements/{req_preset}.txt"

    # Use the venv's interpreter to install the project along with
    # all it's dev dependencies, this ensures it's installed in the right way
    print(f"Setting up development environment from '{requirements_file}'...")
    session.run(
        python_path,
        "-m",
        "pip",
        "install",
        "-r",
        requirements_file,
        "-e.",
        external=True,
    )


@nox.session(reuse_venv=True)
def requirements(session: nox.Session) -> None:
    """Freeze requirements files from project specification and synchronize versions across tools."""  # noqa: W505 [doc-line-too-long]
    requirements_path = ROOT_DIR / "requirements"
    req_sync_tool = requirements_path / "sync_tool.py"

    dependencies = ["pre-commit"] + nox.project.load_toml(req_sync_tool)["dependencies"]
    session.install(*dependencies)
    session.install("pip-compile-multi")

    session.run("python", req_sync_tool, "pull")
    session.run("pip-compile-multi", "-g", "--skip-constraints")
    session.run("python", req_sync_tool, "push")

    session.run("pre-commit", "run", "--files", ".pre-commit-config.yaml", success_codes=[0, 1])


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Regenerate and build all API and user docs."""
    session.notify("api_docs")
    session.notify("user_docs", posargs=session.posargs)


@nox.session(reuse_venv=True)
def api_docs(session: nox.Session) -> None:
    """Build (regenerate) API docs."""
    session.install(f"sphinx=={REQUIREMENTS['sphinx']}")
    session.chdir("docs")
    session.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/jace",
    )


@nox.session(reuse_venv=True)
def user_docs(session: nox.Session) -> None:
    """Build the user docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""  # noqa: W505 [doc-line-too-long]
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []
    session.install("-e", ".", "-r", "requirements/dev.txt", *extra_installs)
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run("sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs)
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    build_path = ROOT_DIR / "build"
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install(f"build=={REQUIREMENTS['build']}")
    session.run("python", "-m", "build")
