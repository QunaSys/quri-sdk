# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence

import stim

from quri_parts.core.estimator import (
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    QuantumEstimator,
)
from quri_parts.core.operator import zero
from quri_parts.core.state.state import CircuitQuantumState
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.stim.circuit import convert_circuit

from ..operator import convert_operator

if TYPE_CHECKING:
    from concurrent.futures import Executor


class _Estimate(NamedTuple):
    value: complex
    error: float = 0.0


def _estimate(operator: Estimatable, state: CircuitQuantumState) -> Estimate[complex]:
    if operator == zero():
        return _Estimate(value=0.0)

    exp_val: complex = 0.0
    qubit_count = state.qubit_count

    sim = stim.TableauSimulator()
    circuit = convert_circuit(state.circuit)
    op_terms = convert_operator(operator, qubit_count)

    sim.do_circuit(circuit)
    generators = sim.canonical_stabilizers()

    for p_string, coef in op_terms:
        if not all(generator.commutes(p_string) for generator in generators):
            continue
        exp_val += coef * sim.peek_observable_expectation(p_string)

    return _Estimate(value=exp_val)


def create_stim_clifford_estimator() -> QuantumEstimator[CircuitQuantumState]:
    """Returns a :class:`~QuantumEstimator` that uses stim's
    :class:`TableauSimulator` to calculate expectation values."""
    return _estimate


def _sequential_estimate(
    _: Any, op_state_tuples: Sequence[tuple[Estimatable, CircuitQuantumState]]
) -> Sequence[Estimate[complex]]:
    return [_estimate(operator, state) for operator, state in op_state_tuples]


def _sequential_estimate_single_state(
    state: CircuitQuantumState, operators: Sequence[Estimatable]
) -> Sequence[Estimate[complex]]:
    qubit_count = state.qubit_count

    sim = stim.TableauSimulator()
    circuit = convert_circuit(state.circuit)
    sim.do_circuit(circuit)
    generators = sim.canonical_stabilizers()

    results = []
    for op in operators:
        stim_op = convert_operator(op, qubit_count)
        exp_val: complex = 0.0
        for p_string, coef in stim_op:
            if not all(generator.commutes(p_string) for generator in generators):
                continue
            exp_val += coef * sim.peek_observable_expectation(p_string)
        results.append(_Estimate(value=exp_val))
    return results


def _concurrent_estimate(
    operators: Sequence[Estimatable],
    states: Sequence[CircuitQuantumState],
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


def create_stim_clifford_concurrent_estimator(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentQuantumEstimator[CircuitQuantumState]:
    """Returns a :class:`~ConcurrentQuantumEstimator` that uses stim's
    :class:`TableauSimulator` to calculate expectation values."""

    def estimator(
        operators: Sequence[Estimatable], states: Sequence[CircuitQuantumState]
    ) -> Sequence[Estimate[complex]]:
        return _concurrent_estimate(operators, states, executor, concurrency)

    return estimator
