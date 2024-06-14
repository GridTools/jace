# Contributing

JaCe is an open-source project that accepts contributions from any individual or organization. Proper credit will be given to contributors by adding their names to the [AUTHORS.md](AUTHORS.md) file.

# Quick development

The fastest way to start with development is to use nox. If you don't have nox, you can use `pipx run nox` to run it without installing, or `pipx install nox`. If you don't have pipx (pip for applications), then you can install with `pip install pipx` (the only case were installing an application with regular pip is reasonable). If you use macOS, then pipx and nox are both in brew, use `brew install pipx nox`.

To use, run `nox`. This will lint and test using every installed version of Python on your system, skipping ones that are not installed. You can also run specific jobs:

```console
$ nox -s lint  # Lint only
$ nox -s tests  # Python tests
$ nox -s docs -- --serve  # Build and serve the docs
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual environment for each run.

# Setting up a development environment manually

You can set up a development environment by running:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements/dev.txt
pip install -v -e .
```

If you have the [Python Launcher for Unix](https://github.com/brettcannon/python-launcher), you can instead do:

```bash
py -m venv .venv
py -m pip install --upgrade pip setuptools wheel
py -m pip install -r requirements/dev.txt
py -m pip install -v -e .
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or `pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=JaCe
```

# Building docs

You can build the docs using:

```bash
nox -s docs
```

You can see a preview with:

```bash
nox -s docs -- --serve
```

# Pre-commit

This project uses pre-commit for all style checking. While you can run it with nox, this is such an important tool that it deserves to be installed on its own. Install pre-commit and run:

```bash
pre-commit run -a
```

to check all files.

# Pull requests (PRs) and merge guidelines

Before submitting a pull request, check that it meets the following criteria:

1. Pull request with code changes should always include tests.
2. If the pull request adds functionality, it should be documented both in the code docstrings and in the official documentation.
3. The pull request should have a proper description of its intent and the main changes in the code. In general this description should be used as commit message if the pull request is approved (check point **5.** below).
4. If the pull request contains code authored by first-time contributors, they should add their names to the [AUTHORS.md](AUTHORS.md) file.
5. Pick one reviewer and try to contact them directly to let them know about the pull request. If there is no feedback in 24h/48h try to contact them again or pick another reviewer.
6. Once the pull request has been approved, it should be squash-merged as soon as possible with a meaningful description of the changes. We use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) specification for writing informative and automation-friendly commit messages. The following _commit types_ are accepted:
   - `build`: changes that affect the build system or external dependencies
   - `chore`: changes related to the development tools or process
   - `ci`: changes to our CI configuration files and scripts
   - `docs`: documentation only changes
   - `feat`: a new feature
   - `fix`: a bug fix
   - `perf`: a code change that improves performance
   - `refactor`: a code change that neither fixes a bug nor adds a feature
   - `style`: changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
   - `test`: adding missing tests or correcting existing tests
