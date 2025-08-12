# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quri_parts.circuit import (
    ImmutableBoundParametricQuantumCircuit,
    ParametricQuantumCircuit,
)
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    ParametricQuantumEstimator,
    create_concurrent_estimator_from_estimator,
    create_concurrent_parametric_estimator_from_concurrent_estimator,
    create_estimator_from_concurrent_estimator,
    create_general_estimator_from_concurrent_estimator,
    create_general_estimator_from_estimator,
    create_parametric_estimator_from_concurrent_estimator,
)
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import (
    CircuitQuantumState,
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)


@dataclass
class _Estimate:
    value: complex
    error: float = 0.0


def fake_estimator(op: Estimatable, state: CircuitQuantumState) -> _Estimate:
    op_contribution = len(op) if isinstance(op, Operator) else 1

    if isinstance(state, ComputationalBasisState):
        return _Estimate(value=state.bits * op_contribution + 0j)
    elif isinstance(state, GeneralCircuitQuantumState) and isinstance(
        state.circuit, ImmutableBoundParametricQuantumCircuit
    ):
        e_list = [v * op_contribution for v in state.circuit.parameter_map.values()]
        return _Estimate(sum(e_list))
    else:
        assert False


def fake_concurrent_estimator(
    operators: Sequence[Estimatable], states: Sequence[CircuitQuantumState]
) -> Sequence[_Estimate]:
    num_ops = len(operators)
    num_states = len(states)

    if num_states == 1:
        states = [next(iter(states))] * num_ops

    if num_ops == 1:
        operators = [next(iter(operators))] * num_states

    return [fake_estimator(op, state) for (op, state) in zip(operators, states)]


def fake_vector_estimator(op: Estimatable, state: QuantumStateVector) -> _Estimate:
    op_contribution = len(op) if isinstance(op, Operator) else 1
    estimate = op_contribution * state.vector.sum()
    if isinstance(state.circuit, ImmutableBoundParametricQuantumCircuit):
        estimate *= sum(state.circuit.parameter_map.values())
    return _Estimate(estimate)


def fake_concurrent_vector_estimator(
    operators: Sequence[Estimatable], states: Sequence[QuantumStateVector]
) -> Sequence[_Estimate]:
    num_ops = len(operators)
    num_states = len(states)

    if num_states == 1:
        states = [next(iter(states))] * num_ops

    if num_ops == 1:
        operators = [next(iter(operators))] * num_states

    return [fake_vector_estimator(op, state) for (op, state) in zip(operators, states)]


class TestCreateConcurrentEstimatorFromEstimator:
    def test_with_circuit_quantum_state(self) -> None:
        concurrent_estimator = create_concurrent_estimator_from_estimator(
            fake_estimator
        )
        op = PAULI_IDENTITY
        state_1 = ComputationalBasisState(1)
        state_2 = ComputationalBasisState(1, bits=1)
        assert concurrent_estimator([op], [state_1]) == [_Estimate(value=0 + 0j)]
        assert concurrent_estimator([op], [state_1, state_2]) == [
            _Estimate(0 + 0j),
            _Estimate(1 + 0j),
        ]
        assert concurrent_estimator([op, op], [state_1]) == [
            _Estimate(0 + 0j),
            _Estimate(0 + 0j),
        ]

        op_1 = Operator({pauli_label("X0"): 1, pauli_label("Y0"): 1})
        assert concurrent_estimator([op, op_1], [state_1, state_2]) == [
            _Estimate(value=0 + 0j),
            _Estimate(value=2 + 0j),
        ]

    def test_with_vector(self) -> None:
        concurrent_estimator = create_concurrent_estimator_from_estimator(
            fake_vector_estimator
        )

        state_0 = QuantumStateVector(1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)])
        state_1 = QuantumStateVector(1, vector=[1 / np.sqrt(2), 1j / np.sqrt(2)])

        op_0 = PAULI_IDENTITY
        op_1 = Operator({pauli_label("X0"): 1, pauli_label("Y0"): 1})

        estimates = list(concurrent_estimator([op_0], [state_0]))
        assert len(estimates) == 1
        assert np.isclose(estimates[0].value, np.sqrt(2))

        estimates = list(concurrent_estimator([op_0], [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, (1 + 1j) / np.sqrt(2))

        estimates = list(concurrent_estimator([op_0, op_1], [state_0]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * np.sqrt(2))

        estimates = list(concurrent_estimator([op_0, op_1], [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * (1 + 1j) / np.sqrt(2))


class TestCreateEstimatorFromConcurrentEstimator:
    def test_with_circuit_quantum_state(self) -> None:
        estimator = create_estimator_from_concurrent_estimator(
            fake_concurrent_estimator
        )
        op = PAULI_IDENTITY
        state = ComputationalBasisState(1, bits=1)
        assert estimator(op, state) == _Estimate(value=1 + 0j)

    def test_with_vector(self) -> None:
        estimator = create_estimator_from_concurrent_estimator(
            fake_concurrent_vector_estimator
        )
        op = PAULI_IDENTITY
        state = QuantumStateVector(1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)])
        assert np.isclose(estimator(op, state).value, np.sqrt(2))


class TestCreateParamtericEstimatorFromConcurrentEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.param_circuit = ParametricQuantumCircuit(1)
        self.param_circuit.add_ParametricRX_gate(0)
        self.param_circuit.add_ParametricRY_gate(0)
        self.param_circuit.add_ParametricRZ_gate(0)

        self.op = PAULI_IDENTITY

    def test_with_circuit_state(self) -> None:
        parametric_estimator: ParametricQuantumEstimator[
            ParametricCircuitQuantumState
        ] = create_parametric_estimator_from_concurrent_estimator(
            fake_concurrent_estimator
        )

        param_state = ParametricCircuitQuantumState(1, self.param_circuit)

        assert parametric_estimator(self.op, param_state, [0, 1, 2]) == _Estimate(
            value=3 + 0j
        )
        assert parametric_estimator(self.op, param_state, [3, 4, 5]) == _Estimate(
            value=12 + 0j
        )

    def test_with_vector(self) -> None:
        parametric_estimator: ParametricQuantumEstimator[
            ParametricQuantumStateVector
        ] = create_parametric_estimator_from_concurrent_estimator(
            fake_concurrent_vector_estimator
        )

        param_state = ParametricQuantumStateVector(
            1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)], circuit=self.param_circuit
        )

        estimate = parametric_estimator(self.op, param_state, [0, 1, 2])
        assert np.isclose(estimate.value, 3 * np.sqrt(2))

        estimate = parametric_estimator(self.op, param_state, [3, 4, 5])
        assert np.isclose(estimate.value, 12 * np.sqrt(2))


class TestCreateConcurrentParamtericEstimatorFromConcurrentEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.param_circuit = ParametricQuantumCircuit(1)
        self.param_circuit.add_ParametricRX_gate(0)
        self.param_circuit.add_ParametricRY_gate(0)
        self.param_circuit.add_ParametricRZ_gate(0)

        self.op = PAULI_IDENTITY

    def test_with_circuit_state(self) -> None:
        concurrent_param_estimator: ConcurrentParametricQuantumEstimator[
            ParametricCircuitQuantumState
        ] = create_concurrent_parametric_estimator_from_concurrent_estimator(
            fake_concurrent_estimator
        )

        param_state = ParametricCircuitQuantumState(1, self.param_circuit)

        assert concurrent_param_estimator(
            self.op, param_state, [[0, 1, 2], [3, 4, 5]]
        ) == [
            _Estimate(value=3 + 0j),
            _Estimate(value=12 + 0j),
        ]

    def test_with_vector(self) -> None:
        concurrent_param_estimator: ConcurrentParametricQuantumEstimator[
            ParametricQuantumStateVector
        ] = create_concurrent_parametric_estimator_from_concurrent_estimator(
            fake_concurrent_vector_estimator
        )

        param_state = ParametricQuantumStateVector(
            1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)], circuit=self.param_circuit
        )

        estimates = concurrent_param_estimator(
            self.op, param_state, [[0, 1, 2], [3, 4, 5]]
        )
        estimate_list = list(estimates)
        assert np.isclose(estimate_list[0].value, 3 * np.sqrt(2))
        assert np.isclose(estimate_list[1].value, 12 * np.sqrt(2))


class TestGeneralQuantumEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.op_0 = PAULI_IDENTITY
        self.op_1 = Operator({pauli_label("X0"): 1, pauli_label("Y0"): 1})

        self.param_circuit = ParametricQuantumCircuit(1)
        self.param_circuit.add_ParametricRX_gate(0)
        self.param_circuit.add_ParametricRY_gate(0)
        self.param_circuit.add_ParametricRZ_gate(0)

    def test_with_circuit_state(self) -> None:
        general_estimator = create_general_estimator_from_estimator(fake_estimator)

        state_0 = ComputationalBasisState(1)
        state_1 = ComputationalBasisState(1, bits=1)

        assert general_estimator(self.op_0, state_0) == _Estimate(value=0 + 0j)
        assert general_estimator(self.op_0, state_1) == _Estimate(value=1 + 0j)

        assert general_estimator([self.op_0], [state_0]) == [_Estimate(value=0 + 0j)]
        assert general_estimator([self.op_0], [state_0, state_1]) == [
            _Estimate(value=0 + 0j),
            _Estimate(value=1 + 0j),
        ]
        assert general_estimator(self.op_0, [state_0, state_1]) == [
            _Estimate(value=0 + 0j),
            _Estimate(value=1 + 0j),
        ]
        assert general_estimator([self.op_0, self.op_1], [state_1]) == [
            _Estimate(value=1 + 0j),
            _Estimate(value=2 + 0j),
        ]
        assert general_estimator([self.op_0, self.op_1], state_1) == [
            _Estimate(value=1 + 0j),
            _Estimate(value=2 + 0j),
        ]
        assert general_estimator([self.op_0, self.op_1], [state_0, state_1]) == [
            _Estimate(value=0 + 0j),
            _Estimate(value=2 + 0j),
        ]

        param_state = ParametricCircuitQuantumState(1, self.param_circuit)

        assert general_estimator(self.op_0, param_state, [0, 1, 2]) == _Estimate(
            value=3 + 0j
        )
        assert general_estimator(self.op_0, param_state, [3, 4, 5]) == _Estimate(
            value=12 + 0j
        )
        assert general_estimator(
            self.op_0, param_state, np.array([0, 1, 2])
        ) == _Estimate(value=3 + 0j)
        assert general_estimator(
            self.op_0, param_state, np.array([3, 4, 5])
        ) == _Estimate(value=12 + 0j)

        assert general_estimator(self.op_0, param_state, [[0, 1, 2], [3, 4, 5]]) == [
            _Estimate(value=3 + 0j),
            _Estimate(value=12 + 0j),
        ]
        assert general_estimator(
            self.op_0, param_state, [np.array([0, 1, 2]), np.array([3, 4, 5])]
        ) == [
            _Estimate(value=3 + 0j),
            _Estimate(value=12 + 0j),
        ]
        assert general_estimator(
            self.op_0, param_state, np.array([[0, 1, 2], [3, 4, 5]])
        ) == [
            _Estimate(value=3 + 0j),
            _Estimate(value=12 + 0j),
        ]

    def test_with_vector(self) -> None:
        general_estimator = create_general_estimator_from_estimator(
            fake_vector_estimator
        )
        state_0 = QuantumStateVector(1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)])
        state_1 = QuantumStateVector(1, vector=[1 / np.sqrt(2), 1j / np.sqrt(2)])

        assert np.isclose(general_estimator(self.op_0, state_0).value, np.sqrt(2))
        assert np.isclose(
            general_estimator(self.op_0, state_1).value, (1 + 1j) / np.sqrt(2)
        )

        estimates = list(general_estimator([self.op_0], [state_0]))
        assert len(estimates) == 1
        assert np.isclose(estimates[0].value, np.sqrt(2))

        estimates = list(general_estimator([self.op_0], [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, (1 + 1j) / np.sqrt(2))

        estimates = list(general_estimator(self.op_0, [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, (1 + 1j) / np.sqrt(2))

        estimates = list(general_estimator([self.op_0, self.op_1], [state_0]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * np.sqrt(2))

        estimates = list(general_estimator([self.op_0, self.op_1], state_0))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * np.sqrt(2))

        estimates = list(general_estimator([self.op_0, self.op_1], [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * (1 + 1j) / np.sqrt(2))

        param_state = ParametricQuantumStateVector(
            1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)], circuit=self.param_circuit
        )

        estimate = general_estimator(self.op_0, param_state, [0, 1, 2])
        assert np.isclose(estimate.value, 3 * np.sqrt(2))
        estimate = general_estimator(self.op_0, param_state, [3, 4, 5])
        assert np.isclose(estimate.value, 12 * np.sqrt(2))
        estimate = general_estimator(self.op_0, param_state, np.array([0, 1, 2]))
        assert np.isclose(estimate.value, 3 * np.sqrt(2))
        estimate = general_estimator(self.op_0, param_state, np.array([3, 4, 5]))
        assert np.isclose(estimate.value, 12 * np.sqrt(2))

        estimates = list(
            general_estimator(self.op_0, param_state, [[0, 1, 2], [3, 4, 5]])
        )
        assert np.isclose(estimates[0].value, 3 * np.sqrt(2))
        assert np.isclose(estimates[1].value, 12 * np.sqrt(2))
        estimates = list(
            general_estimator(
                self.op_0, param_state, [np.array([0, 1, 2]), np.array([3, 4, 5])]
            )
        )
        assert np.isclose(estimates[0].value, 3 * np.sqrt(2))
        assert np.isclose(estimates[1].value, 12 * np.sqrt(2))
        estimates = list(
            general_estimator(self.op_0, param_state, np.array([[0, 1, 2], [3, 4, 5]]))
        )
        assert np.isclose(estimates[0].value, 3 * np.sqrt(2))
        assert np.isclose(estimates[1].value, 12 * np.sqrt(2))

    def test_concurrent_with_circuit_state(self) -> None:
        general_estimator = create_general_estimator_from_concurrent_estimator(
            fake_concurrent_estimator
        )

        state_0 = ComputationalBasisState(1)
        state_1 = ComputationalBasisState(1, bits=1)

        assert general_estimator(self.op_0, state_0) == _Estimate(value=0 + 0j)
        assert general_estimator(self.op_0, state_1) == _Estimate(value=1 + 0j)

        assert general_estimator([self.op_0], [state_0]) == [_Estimate(value=0 + 0j)]
        assert general_estimator([self.op_0], [state_0, state_1]) == [
            _Estimate(value=0 + 0j),
            _Estimate(value=1 + 0j),
        ]
        assert general_estimator(self.op_0, [state_0, state_1]) == [
            _Estimate(value=0 + 0j),
            _Estimate(value=1 + 0j),
        ]
        assert general_estimator([self.op_0, self.op_1], [state_1]) == [
            _Estimate(value=1 + 0j),
            _Estimate(value=2 + 0j),
        ]
        assert general_estimator([self.op_0, self.op_1], state_1) == [
            _Estimate(value=1 + 0j),
            _Estimate(value=2 + 0j),
        ]
        assert general_estimator([self.op_0, self.op_1], [state_0, state_1]) == [
            _Estimate(value=0 + 0j),
            _Estimate(value=2 + 0j),
        ]

        param_state = ParametricCircuitQuantumState(1, self.param_circuit)

        assert general_estimator(self.op_0, param_state, [0, 1, 2]) == _Estimate(
            value=3 + 0j
        )
        assert general_estimator(self.op_0, param_state, [3, 4, 5]) == _Estimate(
            value=12 + 0j
        )
        assert general_estimator(
            self.op_0, param_state, np.array([0, 1, 2])
        ) == _Estimate(value=3 + 0j)
        assert general_estimator(
            self.op_0, param_state, np.array([3, 4, 5])
        ) == _Estimate(value=12 + 0j)

        assert general_estimator(self.op_0, param_state, [[0, 1, 2], [3, 4, 5]]) == [
            _Estimate(value=3 + 0j),
            _Estimate(value=12 + 0j),
        ]
        assert general_estimator(
            self.op_0, param_state, [np.array([0, 1, 2]), np.array([3, 4, 5])]
        ) == [
            _Estimate(value=3 + 0j),
            _Estimate(value=12 + 0j),
        ]
        assert general_estimator(
            self.op_0, param_state, np.array([[0, 1, 2], [3, 4, 5]])
        ) == [
            _Estimate(value=3 + 0j),
            _Estimate(value=12 + 0j),
        ]

    def test_concurrent_with_vector(self) -> None:
        general_estimator = create_general_estimator_from_concurrent_estimator(
            fake_concurrent_vector_estimator
        )
        state_0 = QuantumStateVector(1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)])
        state_1 = QuantumStateVector(1, vector=[1 / np.sqrt(2), 1j / np.sqrt(2)])

        assert np.isclose(general_estimator(self.op_0, state_0).value, np.sqrt(2))
        assert np.isclose(
            general_estimator(self.op_0, state_1).value, (1 + 1j) / np.sqrt(2)
        )

        estimates = list(general_estimator([self.op_0], [state_0]))
        assert len(estimates) == 1
        assert np.isclose(estimates[0].value, np.sqrt(2))

        estimates = list(general_estimator([self.op_0], [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, (1 + 1j) / np.sqrt(2))

        estimates = list(general_estimator(self.op_0, [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, (1 + 1j) / np.sqrt(2))

        estimates = list(general_estimator([self.op_0, self.op_1], [state_0]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * np.sqrt(2))

        estimates = list(general_estimator([self.op_0, self.op_1], state_0))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * np.sqrt(2))

        estimates = list(general_estimator([self.op_0, self.op_1], [state_0, state_1]))
        assert len(estimates) == 2
        assert np.isclose(estimates[0].value, np.sqrt(2))
        assert np.isclose(estimates[1].value, 2 * (1 + 1j) / np.sqrt(2))

        param_state = ParametricQuantumStateVector(
            1, vector=[1 / np.sqrt(2), 1 / np.sqrt(2)], circuit=self.param_circuit
        )

        estimate = general_estimator(self.op_0, param_state, [0, 1, 2])
        assert np.isclose(estimate.value, 3 * np.sqrt(2))
        estimate = general_estimator(self.op_0, param_state, [3, 4, 5])
        assert np.isclose(estimate.value, 12 * np.sqrt(2))
        estimate = general_estimator(self.op_0, param_state, np.array([0, 1, 2]))
        assert np.isclose(estimate.value, 3 * np.sqrt(2))
        estimate = general_estimator(self.op_0, param_state, np.array([3, 4, 5]))
        assert np.isclose(estimate.value, 12 * np.sqrt(2))

        estimates = list(
            general_estimator(self.op_0, param_state, [[0, 1, 2], [3, 4, 5]])
        )
        assert np.isclose(estimates[0].value, 3 * np.sqrt(2))
        assert np.isclose(estimates[1].value, 12 * np.sqrt(2))
        estimates = list(
            general_estimator(
                self.op_0, param_state, [np.array([0, 1, 2]), np.array([3, 4, 5])]
            )
        )
        assert np.isclose(estimates[0].value, 3 * np.sqrt(2))
        assert np.isclose(estimates[1].value, 12 * np.sqrt(2))
        estimates = list(
            general_estimator(self.op_0, param_state, np.array([[0, 1, 2], [3, 4, 5]]))
        )
        assert np.isclose(estimates[0].value, 3 * np.sqrt(2))
        assert np.isclose(estimates[1].value, 12 * np.sqrt(2))
