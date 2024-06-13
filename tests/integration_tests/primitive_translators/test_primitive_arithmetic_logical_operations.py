# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for `MappedOperationTranslatorBase` class and arithmetic & logical operations.

The `MappedOperationTranslatorBase` can not be tested on its own, since it does
not generate a Tasklet. For that reason it is thoroughly tested together with
the arithmetic and logical translators (ALT).

Thus the first tests tests the behaviour of the `MappedOperationTranslatorBase`
class such as
- broadcasting,
- literal substitution,
- scalar vs array computation.

Followed by tests that are specific to the ALTs, which mostly focuses
on the validity of the template of the ALT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dace
import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jace

from tests import util as testutil


if TYPE_CHECKING:
    from collections.abc import Callable, Generator


@pytest.fixture(autouse=True)
def _only_alt_translators() -> Generator[None, None, None]:
    """Removes all non arithmetic/logical translator from the registry.

    This ensures that Jax is not doing some stuff that is supposed to be handled by the
    test class, such as broadcasting. It makes writing tests a bit harder, but it is
    worth. For some reasons also type conversion s allowed.
    """
    from jace.translator.primitive_translators.arithmetic_logical_translators import (  # noqa: PLC0415  # Direct import.
        _ARITMETIC_OPERATION_TEMPLATES,  # noqa: PLC2701  # Import of private variables.
        _LOGICAL_OPERATION_TEMPLATES,  # noqa: PLC2701
    )

    # Remove all non ALU translators from the registry
    primitive_translators = jace.translator.get_registered_primitive_translators()
    allowed_translators = (
        _LOGICAL_OPERATION_TEMPLATES.keys()
        | _ARITMETIC_OPERATION_TEMPLATES.keys()
        | {"convert_element_type"}
    )
    testutil.set_active_primitive_translators_to({
        p: t for p, t in primitive_translators.items() if p in allowed_translators
    })

    yield

    # Restore the initial state
    testutil.set_active_primitive_translators_to(primitive_translators)


@pytest.fixture(
    params=[
        (jnp.logical_and, 2, np.bool_),
        (jnp.logical_or, 2, np.bool_),
        (jnp.logical_xor, 2, np.bool_),
        (jnp.logical_not, 1, np.bool_),
        (jnp.bitwise_and, 2, np.int64),
        (jnp.bitwise_or, 2, np.int64),
        (jnp.bitwise_xor, 2, np.int64),
        (jnp.bitwise_not, 1, np.int64),
    ]
)
def logical_ops(request) -> tuple[Callable, tuple[np.ndarray, ...]]:
    """Returns a logical operation function and inputs."""
    return (
        request.param[0],
        tuple(testutil.mkarray((2, 2), request.param[2]) for _ in range(request.param[1])),
    )


@pytest.fixture(
    params=[
        np.float32,
        pytest.param(
            np.complex64,
            marks=pytest.mark.skip("Some complex values operations are not fully supported."),
        ),
    ]
)
def dtype(request) -> type:
    """Data types that should be used for the numerical tests of the ALT translators."""
    return request.param


@pytest.fixture(
    params=[
        lambda x: +(x - 0.5),
        lambda x: -x,
        jnp.floor,
        jnp.ceil,
        jnp.round,
        jnp.exp2,
        lambda x: jnp.abs(-x),
        lambda x: jnp.sqrt(x**2),  # includes integer power.
        lambda x: jnp.log(jnp.exp(x)),
        lambda x: jnp.log1p(jnp.expm1(x)),
        lambda x: jnp.asin(jnp.sin(x)),
        lambda x: jnp.acos(jnp.cos(x)),
        lambda x: jnp.atan(jnp.tan(x)),
        lambda x: jnp.asinh(jnp.sinh(x)),
        lambda x: jnp.acosh(jnp.cosh(x)),
        lambda x: jnp.atanh(jnp.tanh(x)),
    ]
)
def alt_unary_ops(request, dtype: type) -> tuple[Callable, np.ndarray]:
    """The inputs and the operation we need for the full test.

    Some of the unary operations are combined to ensure that they will succeed.
    An example is `asin()` which only takes values in the range `[-1, 1]`.
    """
    return (request.param, testutil.mkarray((2, 2), dtype))


@pytest.fixture(
    params=[
        jnp.add,
        jnp.multiply,
        jnp.divide,
        jnp.minimum,
        jnp.maximum,
        jnp.atan2,
        jnp.nextafter,
        lambda x, y: x**y,
    ]
)
def alt_binary_ops_float(request) -> tuple[Callable, tuple[np.ndarray, np.ndarray]]:
    """Binary ALT operations that operates on floats."""
    # Getting 0 in the division test is unlikely.
    return (  # type: ignore[return-value]  # Type confusion.
        request.param,
        tuple(testutil.mkarray((2, 2), np.float64) for _ in range(2)),
    )


@pytest.fixture(
    params=[
        lambda x, y: x == y,
        lambda x, y: x != y,
        lambda x, y: x <= y,
        lambda x, y: x < y,
        lambda x, y: x >= y,
        lambda x, y: x > y,
    ]
)
def alt_binary_compare_ops(request) -> tuple[Callable, tuple[np.ndarray, np.ndarray]]:
    """Comparison operations, operates on integers."""
    return (
        request.param,
        tuple(np.abs(testutil.mkarray((20, 20), np.int32)) % 30 for _ in range(2)),
    )


@pytest.fixture(
    params=[
        [(100, 1), (100, 10)],
        [(100, 1, 3), (100, 1, 1)],
        [(5, 1, 3, 4, 1, 5), (5, 1, 3, 1, 2, 5)],
    ]
)
def broadcast_input(request) -> tuple[np.ndarray, np.ndarray]:
    """Inputs to be used for the broadcast test."""
    return tuple(testutil.mkarray(shape) for shape in request.param)  # type: ignore[return-value] # can not deduce that it is only size 2.


def _perform_alt_test(testee: Callable, *args: Any) -> None:
    """General function that just performs the test."""
    wrapped = jace.jit(testee)

    ref = testee(*args)
    res = wrapped(*args)

    if jace.util.is_scalar(ref):
        # Builder hack, only arrays are generated.
        assert res.shape == (1,)
    elif ref.shape == ():
        assert res.shape == (1,)
    else:
        assert ref.shape == res.shape
    assert ref.dtype == res.dtype
    assert np.allclose(ref, res), f"Expected '{ref.tolist()}' got '{res.tolist()}'"


# <------------ Tests for `MappedOperationTranslatorBase`


def test_mapped_unary_scalar() -> None:
    def testee(A: np.float64) -> np.float64 | jax.Array:
        return jnp.cos(A)

    _perform_alt_test(testee, np.float64(1.0))


def test_mapped_unary_array() -> None:
    def testee(A: np.ndarray) -> jax.Array:
        return jnp.sin(A)

    A = testutil.mkarray((100, 10, 3))

    _perform_alt_test(testee, A)


def test_mapped_unary_scalar_literal() -> None:
    def testee(A: float) -> float | jax.Array:
        return jnp.sin(1.98) + A

    _perform_alt_test(testee, 10.0)


def test_mapped_binary_scalar() -> None:
    def testee(A: np.float64, B: np.float64) -> np.float64:
        return A * B

    _perform_alt_test(testee, np.float64(1.0), np.float64(2.0))


def test_mapped_binary_scalar_partial_literal() -> None:
    def testeeR(A: np.float64) -> np.float64:
        return A * 2.03

    def testeeL(A: np.float64) -> np.float64:
        return 2.03 * A

    A = np.float64(7.0)
    _perform_alt_test(testeeR, A)
    _perform_alt_test(testeeL, A)


def test_mapped_binary_array() -> None:
    """Test binary of arrays, with same size."""

    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = testutil.mkarray((100, 10, 3))
    B = testutil.mkarray((100, 10, 3))
    _perform_alt_test(testee, A, B)


def test_mapped_binary_array_scalar() -> None:
    def testee(A: np.ndarray | np.float64, B: np.float64 | np.ndarray) -> np.ndarray:
        return A + B  # type: ignore[return-value]  # It is always an array.

    A = testutil.mkarray((100, 22))
    B = np.float64(1.34)
    _perform_alt_test(testee, A, B)
    _perform_alt_test(testee, B, A)


def test_mapped_binary_array_partial_literal() -> None:
    def testeeR(A: np.ndarray) -> np.ndarray:
        return A + 1.52

    def testeeL(A: np.ndarray) -> np.ndarray:
        return 1.52 + A

    A = testutil.mkarray((100, 22))
    _perform_alt_test(testeeR, A)
    _perform_alt_test(testeeL, A)


def test_mapped_binary_array_constants() -> None:
    def testee(A: np.ndarray) -> np.ndarray:
        return A + jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    A = testutil.mkarray((3, 3))
    _perform_alt_test(testee, A)


def test_mapped_broadcast(broadcast_input: tuple[np.ndarray, np.ndarray]) -> None:
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A + B

    A = broadcast_input[0]
    B = broadcast_input[1]
    _perform_alt_test(testee, A, B)
    _perform_alt_test(testee, B, A)


# <------------ Tests for arithmetic and logical translators/operations


def test_alt_general_unary(alt_unary_ops: tuple[Callable, np.ndarray]) -> None:
    def testee(A: np.ndarray) -> np.ndarray:
        return alt_unary_ops[0](A)

    _perform_alt_test(testee, alt_unary_ops[1])


def test_alt_unary_isfinite() -> None:
    def testee(A: np.ndarray) -> jax.Array:
        return jnp.isfinite(A)

    A = np.array([np.inf, +np.inf, -np.inf, np.nan, -np.nan, 1.0])

    args = dace.Config.get("compiler", "cpu", "args")
    try:
        new_args = args.replace("-ffast-math", "-fno-finite-math-only")
        dace.Config.set("compiler", "cpu", "args", value=new_args)
        _perform_alt_test(testee, A)

    finally:
        dace.Config.set("compiler", "cpu", "args", value=args)


def test_alt_general_binary_float(
    alt_binary_ops_float: tuple[Callable, tuple[np.ndarray, np.ndarray]],
) -> None:
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return alt_binary_ops_float[0](A, B)

    _perform_alt_test(testee, *alt_binary_ops_float[1])


def test_alt_compare_operation(
    alt_binary_compare_ops: tuple[Callable, tuple[np.ndarray, np.ndarray]],
) -> None:
    def testee(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return alt_binary_compare_ops[0](A, B)

    _perform_alt_test(testee, *alt_binary_compare_ops[1])


def test_alt_logical_bitwise_operation(
    logical_ops: tuple[Callable, tuple[np.ndarray, ...]],
) -> None:
    inputs: tuple[np.ndarray, ...] = logical_ops[1]

    def testee(*args: np.ndarray) -> np.ndarray:
        return logical_ops[0](*args)

    _perform_alt_test(testee, *inputs)


def test_alt_unary_integer_power() -> None:
    def testee(A: np.ndarray) -> np.ndarray:
        return A**3

    A = testutil.mkarray((10, 2, 3))
    _perform_alt_test(testee, A)
