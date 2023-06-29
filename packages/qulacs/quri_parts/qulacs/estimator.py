# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Union

import qulacs

from quri_parts.circuit.noise import NoiseModel
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
from quri_parts.core.state import ParametricQuantumStateVector, QuantumStateVector
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.qulacs import QulacsParametricStateT, QulacsStateT
from quri_parts.qulacs.circuit.compiled_circuit import (
    _QulacsCircuit,
    _QulacsLinearMappedUnboundParametricCircuit,
    _QulacsUnboundParametricCircuit,
)

from . import cast_to_list
from .circuit import convert_circuit, convert_parametric_circuit
from .circuit.noise import convert_circuit_with_noise_model
from .operator import convert_operator

if TYPE_CHECKING:
    from concurrent.futures import Executor


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


def _create_qulacs_initial_state(
    state: Union[QulacsStateT, QulacsParametricStateT]
) -> qulacs.QuantumState:
    qs_state = qulacs.QuantumState(state.qubit_count)
    if isinstance(state, (QuantumStateVector, ParametricQuantumStateVector)):
        qs_state.load(cast_to_list(state.vector))
    return qs_state


def _estimate(operator: Estimatable, state: QulacsStateT) -> Estimate[complex]:
    if operator == zero():
        return _Estimate(value=0.0)
    if isinstance(state.circuit, _QulacsCircuit):
        circuit = state.circuit._qulacs_circuit
    else:
        circuit = convert_circuit(state.circuit)
    qs_state = _create_qulacs_initial_state(state)
    op = convert_operator(operator, state.qubit_count)
    circuit.update_quantum_state(qs_state)
    exp = op.get_expectation_value(qs_state)
    return _Estimate(value=exp)


def create_qulacs_vector_estimator() -> QuantumEstimator[QulacsStateT]:
    """Returns a :class:`~QuantumEstimator` that uses Qulacs vector simulator
    to calculate expectation values."""

    return _estimate


def _sequential_estimate(
    _: Any, op_state_tuples: Sequence[tuple[Estimatable, QulacsStateT]]
) -> Sequence[Estimate[complex]]:
    return [_estimate(operator, state) for operator, state in op_state_tuples]


def _sequential_estimate_single_state(
    state: QulacsStateT, operators: Sequence[Estimatable]
) -> Sequence[Estimate[complex]]:
    if isinstance(state.circuit, _QulacsCircuit):
        circuit = state.circuit._qulacs_circuit
    else:
        circuit = convert_circuit(state.circuit)
    n_qubits = state.qubit_count
    qs_state = _create_qulacs_initial_state(state)
    circuit.update_quantum_state(qs_state)
    results = []
    for op in operators:
        qs_op = convert_operator(op, n_qubits)
        results.append(_Estimate(value=qs_op.get_expectation_value(qs_state)))
    return results


def _concurrent_estimate(
    _sequential_estimate: Callable[
        [Any, Sequence[tuple[Estimatable, QulacsStateT]]], Sequence[Estimate[complex]]
    ],
    _sequential_estimate_single_state: Callable[
        [QulacsStateT, Sequence[Estimatable]], Sequence[Estimate[complex]]
    ],
    operators: Collection[Estimatable],
    states: Collection[QulacsStateT],
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


def create_qulacs_vector_concurrent_estimator(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentQuantumEstimator[QulacsStateT]:
    """Returns a :class:`~ConcurrentQuantumEstimator` that uses Qulacs vector
    simulator to calculate expectation values."""

    def estimator(
        operators: Collection[Estimatable],
        states: Collection[QulacsStateT],
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
    op_state: tuple[Estimatable, QulacsParametricStateT],
    params: Sequence[Sequence[float]],
) -> Sequence[Estimate[complex]]:
    operator, state = op_state
    n_qubits = state.qubit_count
    op = convert_operator(operator, n_qubits)
    parametric_circuit = state.parametric_circuit
    if isinstance(
        parametric_circuit,
        (_QulacsLinearMappedUnboundParametricCircuit, _QulacsUnboundParametricCircuit),
    ):
        qulacs_circuit, param_mapper = (
            parametric_circuit.qulacs_circuit,
            parametric_circuit.param_mapper,
        )
    else:
        qulacs_circuit, param_mapper = convert_parametric_circuit(parametric_circuit)

    estimates = []
    for param in params:
        for i, v in enumerate(param_mapper(param)):
            qulacs_circuit.set_parameter(i, v)
        qs_state = _create_qulacs_initial_state(state)
        qulacs_circuit.update_quantum_state(qs_state)
        exp = op.get_expectation_value(qs_state)
        estimates.append(_Estimate(value=exp))

    return estimates


def create_qulacs_vector_parametric_estimator() -> (
    ParametricQuantumEstimator[QulacsParametricStateT]
):
    def estimator(
        operator: Estimatable, state: QulacsParametricStateT, param: Sequence[float]
    ) -> Estimate[complex]:
        ests = _sequential_parametric_estimate((operator, state), [param])
        return ests[0]

    return estimator


def create_qulacs_vector_concurrent_parametric_estimator(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentParametricQuantumEstimator[QulacsParametricStateT]:
    def estimator(
        operator: Estimatable,
        state: QulacsParametricStateT,
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


def create_qulacs_density_matrix_estimator(
    model: NoiseModel,
) -> QuantumEstimator[QulacsStateT]:
    """Returns a :class:~~QuantumEstimator` that uses Qulacs simulator using
    density matrix with noise model."""

    def _estimate_with_noise(
        operator: Estimatable, state: QulacsStateT
    ) -> Estimate[complex]:
        if operator == zero():
            return _Estimate(value=0.0)
        circuit = convert_circuit_with_noise_model(state.circuit, model)
        qs_state = qulacs.DensityMatrix(state.qubit_count)
        if isinstance(state, QuantumStateVector):
            qs_state.load(cast_to_list(state.vector))
        op = convert_operator(operator, state.qubit_count)
        circuit.update_quantum_state(qs_state)
        exp = op.get_expectation_value(qs_state)
        return _Estimate(value=exp)

    return _estimate_with_noise


def create_qulacs_density_matrix_parametric_estimator(
    model: NoiseModel,
) -> ParametricQuantumEstimator[QulacsParametricStateT]:
    return create_parametric_estimator(create_qulacs_density_matrix_estimator(model))


def create_qulacs_density_matrix_concurrent_estimator(
    model: NoiseModel, executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentQuantumEstimator[QulacsStateT]:
    """Returns a :class:`~ConcurrentQuantumEstimator` that uses Qulacs
    simulator using density matrix with noise model to calculate expectation
    values."""

    dm_estimator = create_qulacs_density_matrix_estimator(model)

    def _estimate_sequentially(
        _: Any, op_state_tuples: Sequence[tuple[Estimatable, QulacsStateT]]
    ) -> Sequence[Estimate[complex]]:
        return [dm_estimator(operator, state) for operator, state in op_state_tuples]

    def _estimate_single_state_sequentially(
        state: QulacsStateT, operators: Sequence[Estimatable]
    ) -> Sequence[Estimate[complex]]:
        circuit = convert_circuit_with_noise_model(state.circuit, model)
        qubit_count = state.qubit_count
        qs_state = qulacs.DensityMatrix(qubit_count)
        if isinstance(state, QuantumStateVector):
            qs_state.load(cast_to_list(state.vector))
        circuit.update_quantum_state(qs_state)
        results = []
        for op in operators:
            qs_op = convert_operator(op, qubit_count)
            results.append(_Estimate(value=qs_op.get_expectation_value(qs_state)))
        return results

    def estimator(
        operators: Collection[Estimatable],
        states: Collection[QulacsStateT],
    ) -> Iterable[Estimate[complex]]:
        return _concurrent_estimate(
            _estimate_sequentially,
            _estimate_single_state_sequentially,
            operators,
            states,
            executor,
            concurrency,
        )

    return estimator


def create_qulacs_density_matrix_concurrent_parametric_estimator(
    model: NoiseModel, executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentParametricQuantumEstimator[QulacsParametricStateT]:
    """Returns a :class:`~ConcurrentParametricQuantumEstimator` that uses
    Qulacs simulator using density matrix with noise model to calculate
    expectation values."""

    def _estimate_sequentially(
        op_state: tuple[Estimatable, QulacsParametricStateT],
        params: Sequence[Sequence[float]],
    ) -> Sequence[Estimate[complex]]:
        operator, state = op_state
        qubit_count = state.qubit_count
        op = convert_operator(operator, qubit_count)
        parametric_circuit = state.parametric_circuit

        estimates = []
        for param in params:
            circuit = parametric_circuit.bind_parameters(param)
            qs_circuit = convert_circuit_with_noise_model(circuit, model)
            qs_state = qulacs.DensityMatrix(qubit_count)
            if isinstance(state, (QuantumStateVector, ParametricQuantumStateVector)):
                qs_state.load(cast_to_list(state.vector))
            qs_circuit.update_quantum_state(qs_state)
            exp = op.get_expectation_value(qs_state)
            estimates.append(_Estimate(value=exp))

        return estimates

    def estimator(
        operator: Estimatable,
        state: QulacsParametricStateT,
        params: Sequence[Sequence[float]],
    ) -> Sequence[Estimate[complex]]:
        return execute_concurrently(
            _estimate_sequentially,
            (operator, state),
            params,
            executor,
            concurrency,
        )

    return estimator
