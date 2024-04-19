# Coding Guidelines

## Code Style

We follow the [Google Python Style Guide][google-style-guide] with a few minor changes (mentioned below). Since the best way to remember something is to understand the reasons behind it, make sure you go through the style guide at least once, paying special attention to the discussions in the _Pros_, _Cons_, and _Decision_ subsections.

We deviate from the [Google Python Style Guide][google-style-guide] only in the following points:

- We use [`ruff-linter`][ruff-linter] instead of [`pylint`][pylint].
- We use [`ruff-formatter`][ruff-formatter] for source code and imports formatting, which may work differently than indicated by the guidelines in section [_3. Python Style Rules_](https://google.github.io/styleguide/pyguide.html#3-python-style-rules). For example, maximum line length is set to 100 instead of 79 (although docstring lines should still be limited to 79).
- According to subsection [_2.19 Power Features_](https://google.github.io/styleguide/pyguide.html#219-power-features), direct use of _power features_ (e.g. custom metaclasses, import hacks, reflection) should be avoided, but standard library classes that internally use these power features are accepted. Following the same spirit, we allow the use of power features in infrastructure code with similar functionality and scope as the Python standard library.
- For readability purposes, when a docstring contains more than the required summary line, we prefer indenting the first line at the same cursor position as the first opening quote, although this is not explicitly considered in the doctring conventions described in subsection [_3.8.1 Docstrings_](https://google.github.io/styleguide/pyguide.html#381-docstrings). Example:

  ```python
  # single line docstring
  """A one-line summary of the module or program, terminated by a period."""

  # multi-line docstring
  """
  A one-line summary of the module or program, terminated by a period.

  Leave one blank line. The rest of this docstring should contain an
  overall description of the module or program.
  """
  ```

- According to subsection [_3.19.12 Imports For Typing_](https://google.github.io/styleguide/pyguide.html#31912-imports-for-typing), symbols from `typing` and `collections.abc` modules used in type annotations _"can be imported directly to keep common annotations concise and match standard typing practices"_. Following the same spirit, we allow symbols to be imported directly from third-party or internal modules when they only contain a collection of frequently used typying definitions.

### Common questions

- `pass` vs `...` (`Ellipsis`)

  `pass` is the _no-op_ statement in Python and `...` is a literal value (called _Ellipsis_) introduced for slicing collections of unknown number of dimensions. Although they are very different in nature, both of them are used in places where a statement is required purely for syntactic reasons, and there is not yet a clear standard practice in the community about when to use one or the other. We decided to align with the common pattern of using `...` in the body of empty function definitions working as placeholders for actual implementations defined somewhere else (e.g. type stubs, abstract methods and methods appearing in `Protocol` classes) and `pass` in any other place where its usage is mixed with actual statements.

  ```python
  # Correct use of `...` as the empty body of an abstract method
  class AbstractFoo:
     @abstractmethod
     def bar(self) -> Bar:
        ...

  # Correct use of `pass` when mixed with other statements
  try:
     resource.load(id=42)
  except ResourceException:
     pass
  ```

### Error messages

Error messages should be written as sentences, starting with a capital letter and ending with a period (avoid exclamation marks). Try to be informative without being verbose. Code objects such as 'ClassNames' and 'function_names' should be enclosed in single quotes, and so should string values used for message interpolation.

Examples:

```python
raise ValueError(f"Invalid argument 'dimension': should be of type 'Dimension', got '{dimension.type}'.")
```

Interpolated integer values do not need double quotes, if they are indicating an amount. Example:

```python
raise ValueError(f"Invalid number of arguments: expected 3 arguments, got {len(args)}.")
```

The double quotes can also be dropped when presenting a sequence of values. In this case the message should be rephrased so the sequence is separated from the text by a colon ':'.

```python
raise ValueError(f"unexpected keyword arguments: {', '.join(set(kwarg_names) - set(expected_kwarg_names))}.")
```

The message should be kept to one sentence if reasonably possible. Ideally the sentence should be kept short and avoid unnecessary words. Examples:

```python
# too many sentences
raise ValueError(f"Received an unexpected number of arguments. Should receive 5 arguments, but got {len(args)}. Please provide the correct number of arguments.")
# better
raise ValueError(f"Wrong number of arguments: expected 5, got {len(args)}.")

# less extreme
raise TypeError(f"Wrong argument type. Can only accept 'int's, got '{type(arg)}' instead.")
# but can still be improved
raise TypeError(f"Wrong argument type: 'int' expected, got '{type(arg)}'")
```

The terseness vs. helpfulness tradeoff should be more in favor of terseness for internal error messages and more in favor of helpfulness for `DSLError` and it's subclassses, where additional sentences are encouraged if they point out likely hidden sources of the problem or common fixes.

### Docstrings

TODO: update to `autodoc2`

We generate the API documentation automatically from the docstrings using [Sphinx][sphinx] and some extensions such as [Sphinx-autodoc][sphinx-autodoc] and [Sphinx-napoleon][sphinx-napoleon]. These follow the Google Python Style Guide docstring conventions to automatically format the generated documentation. A complete overview can be found here: [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google).

Sphinx supports the [reStructuredText][sphinx-rest] (reST) markup language for defining additional formatting options in the generated documentation, however section [_3.8 Comments and Docstrings_](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) of the Google Python Style Guide does not specify how to use markups in docstrings. As a result, we decided to forbid reST markup in docstrings, except for the following cases:

- Cross-referencing other objects using Sphinx text roles for the [Python domain](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain) (as explained [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles)).
- Very basic formatting markup to improve _readability_ of the generated documentation without obscuring the source docstring (e.g. ` ``literal`` ` strings, bulleted lists).

We highly encourage the [doctest][doctest] format for code examples in docstrings. In fact, doctest runs code examples and makes sure they are in sync with the codebase.

### Module structure

In general, you should structure new Python modules in the following way:

1. _shebang_ line: `#! /usr/bin/env python3` (only for **executable scripts**!).
2. License header (see `LICENSE_HEADER.txt`).
3. Module docstring.
4. Imports, alphabetically ordered within each block (fixed automatically by `ruff-formatter`):
   1. Block of imports from the standard library.
   2. Block of imports from general third party libraries using standard shortcuts when customary (e.g. `numpy as np`).
   3. Block of imports from specific modules of the project.
5. Definition of exported symbols (optional, mainly for re-exporting symbols from other modules):

```python
__all__ = ["func_a", "CONST_B"]
```

6. Public constants and typing definitions.
7. Module contents organized in a convenient way for understanding how the pieces of code fit together, usually defining functions before classes.

Try to keep sections and items logically ordered, add section separator comments to make section boundaries explicit when needed. If there is not a single evident logical order, pick the order you consider best or use alphabetical order.

Consider configuration files as another type of source code and apply the same criteria, using comments when possible for better readability.

### Ignoring QA errors

You may occasionally need to disable checks from _quality assurance_ (QA) tools (e.g. linters, type checkers, etc.) on specific lines as some tool might not be able to fully understand why a certain piece of code is needed. This is usually done with special comments, e.g. `# noqa: F401`, `# type: ignore`. However, you should **only** ignore QA errors when you fully understand their source and rewriting your code to pass QA checks would make it less readable. Additionally, you should add a short descriptive code if possible (check [ruff rules][ruff-rules] and [mypy error codes][mypy-error-codes] for reference):

```python
f = lambda: 'empty'  # noqa: E731 [lambda-assignment]
```

and, if needed, a brief comment for future reference:

```python
...
return undeclared_symbol  # noqa: F821 [undefined-name] on purpose to trigger black-magic
```

## Testing

Testing components is a critical part of a software development project. We follow standard practices in software development and write unit, integration, and regression tests. Note that even though [doctests][doctest] are great for documentation purposes, they lack many features and are difficult to debug. Hence, they should not be used as replacement for proper unit tests except in trivial cases.

<!-- Reference links -->

[doctest]: https://docs.python.org/3/library/doctest.html
[google-style-guide]: https://google.github.io/styleguide/pyguide.html
[mypy]: https://mypy.readthedocs.io/
[mypy-error-codes]: https://mypy.readthedocs.io/en/stable/error_code_list.html
[pre-commit]: https://pre-commit.com/
[pylint]: https://pylint.pycqa.org/
[ruff-formatter]: https://docs.astral.sh/ruff/formatter/
[ruff-linter]: https://docs.astral.sh/ruff/linter/
[ruff-rules]: https://docs.astral.sh/ruff/rules/
[sphinx]: https://www.sphinx-doc.org
[sphinx-autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[sphinx-napoleon]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#
[sphinx-rest]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
