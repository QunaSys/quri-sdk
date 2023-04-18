from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

import juliacall
import numpy as np
from juliacall import Main as jl
from typing_extensions import TypeAlias

from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    ParametricQuantumEstimator,
    QuantumEstimator,
    create_parametric_estimator,
)
from quri_parts.core.operator import zero
from quri_parts.core.state import CircuitQuantumState, ParametricCircuitQuantumState
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.itensor.load_itensor import ensure_itensor_loaded

from .circuit import convert_circuit
from .operator import convert_operator

if TYPE_CHECKING:
    from concurrent.futures import Executor


class _Estimate(NamedTuple):
    value: complex
    error: float = np.nan


#: A type alias for state classes supported by ITensor estimators.
#: ITensor estimators support circuit states.
ITensorStateT: TypeAlias = CircuitQuantumState

#: A type alias for parametric state classes supported by ITensor estimators.
#: ITensor estimators support circuit states.
ITensorParametricStateT: TypeAlias = ParametricCircuitQuantumState


def _estimate(operator: Estimatable, state: ITensorStateT) -> Estimate[complex]:
    ensure_itensor_loaded()
    if operator == zero():
        return _Estimate(value=0.0, error=0.0)
    qubits = state.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)

    # create ITensor circuit
    circuit = convert_circuit(state.circuit, s)

    # create ITensor operator
    op = convert_operator(operator, s)

    # calculate expectation value
    psi = jl.apply(circuit, psi)
    exp: float = jl.expectation(psi, op)

    return _Estimate(value=exp, error=0.0)


def create_itensor_mps_estimator() -> QuantumEstimator[ITensorStateT]:
    """Returns a :class:`~QuantumEstimator` that uses ITensor MPS simulator to
    calculate expectation values."""

    return _estimate


def _sequential_estimate(
    _: Any, op_state_tuples: Sequence[tuple[Estimatable, ITensorStateT]]
) -> Sequence[Estimate[complex]]:
    return [_estimate(operator, state) for operator, state in op_state_tuples]


def _sequential_estimate_single_state(
    state: ITensorStateT, operators: Sequence[Estimatable]
) -> Sequence[Estimate[complex]]:
    ensure_itensor_loaded()
    qubits = state.qubit_count
    s: juliacall.VectorValue = jl.siteinds("Qubit", qubits)
    psi: juliacall.AnyValue = jl.init_state(s, qubits)
    circuit = convert_circuit(state.circuit, s)
    psi = jl.apply(circuit, psi)
    results = []
    for op in operators:
        if op == zero():
            results.append(_Estimate(value=0.0, error=0.0))
            continue
        itensor_op = convert_operator(op, s)
        results.append(_Estimate(value=jl.expectation(psi, itensor_op), error=0.0))
    return results


def _concurrent_estimate(
    _sequential_estimate: Callable[
        [Any, Sequence[tuple[Estimatable, ITensorStateT]]], Sequence[Estimate[complex]]
    ],
    _sequential_estimate_single_state: Callable[
        [ITensorStateT, Sequence[Estimatable]], Sequence[Estimate[complex]]
    ],
    operators: Collection[Estimatable],
    states: Collection[ITensorStateT],
    executor: Optional["Executor"],
    concurrency: int = 1,
) -> Sequence[Estimate[complex]]:
    num_ops = len(operators)
    num_states = len(states)

    if num_ops == 0:
        raise ValueError("No operator specified.")

    if num_states == 0:
        raise ValueError("No state specified.")

    if num_ops > 1 and num_states > 1 and num_ops != num_states:
        raise ValueError(
            f"Number of operators ({num_ops}) does not match"
            f"number of states ({num_states})."
        )

    if num_states == 1:
        return execute_concurrently(
            _sequential_estimate_single_state,
            next(iter(states)),
            operators,
            executor,
            concurrency,
        )
    else:
        if num_ops == 1:
            operators = [next(iter(operators))] * num_states
        return execute_concurrently(
            _sequential_estimate, None, zip(operators, states), executor, concurrency
        )


def create_itensor_mps_concurrent_estimator(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentQuantumEstimator[ITensorStateT]:
    """Returns a :class:`~ConcurrentQuantumEstimator` that uses ITensor MPS
    simulator to calculate expectation values.

    For now, this function works when the executor is defined like below::

    >>> with ProcessPoolExecutor(
    ...     max_workers=2, mp_context=get_context("spawn")
    ... ) as executor:
    """

    def estimator(
        operators: Collection[Estimatable],
        states: Collection[ITensorStateT],
    ) -> Iterable[Estimate[complex]]:
        return _concurrent_estimate(
            _sequential_estimate,
            _sequential_estimate_single_state,
            operators,
            states,
            executor,
            concurrency,
        )

    return estimator


def _sequential_parametric_estimate(
    op_state: tuple[Estimatable, ITensorParametricStateT],
    params: Sequence[Sequence[float]],
) -> Sequence[Estimate[complex]]:
    operator, state = op_state
    estimates = []
    estimator = create_parametric_estimator(create_itensor_mps_estimator())
    for param in params:
        estimates.append(estimator(operator, state, param))
    return estimates


def create_itensor_mps_parametric_estimator() -> (
    ParametricQuantumEstimator[ITensorParametricStateT]
):
    return create_parametric_estimator(create_itensor_mps_estimator())


def create_itensor_mps_concurrent_parametric_estimator(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentParametricQuantumEstimator[ITensorParametricStateT]:
    def estimator(
        operator: Estimatable,
        state: ITensorParametricStateT,
        params: Sequence[Sequence[float]],
    ) -> Sequence[Estimate[complex]]:
        return execute_concurrently(
            _sequential_parametric_estimate,
            (operator, state),
            params,
            executor,
            concurrency,
        )

    return estimator
