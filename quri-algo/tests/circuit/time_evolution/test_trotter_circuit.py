# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import pytest
from quri_parts.circuit import QuantumCircuit, inverse_circuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator, PauliLabel, pauli_label

from quri_algo.circuit.time_evolution.trotter_time_evo import (
    FixedIntervalTrotterControlledTimeEvolutionCircuitFactory,
    FixedIntervalTrotterTimeEvolutionCircuitFactory,
    FixedStepTrotterControlledTimeEvolutionCircuitFactory,
    FixedStepTrotterTimeEvolutionCircuitFactory,
    TrotterControlledTimeEvolutionCircuitFactory,
    TrotterTimeEvolutionCircuitFactory,
    get_trotter_time_evolution_operator,
)
from quri_algo.problem import QubitHamiltonian


def test_get_trotter_time_evolution_operator() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 3, PAULI_IDENTITY: 2}
    )

    # Trotter order 1
    trotter_unbound_circuit = get_trotter_time_evolution_operator(
        operator, n_state_qubits=2, n_trotter=1
    )
    trotter_circuit = trotter_unbound_circuit.bind_parameters([2.0])
    reconstructed_op = Operator({})
    for g in trotter_circuit.gates:
        reconstructed_op.add_term(
            PauliLabel.from_index_and_pauli_list(g.target_indices, g.pauli_ids),
            0.5 * g.params[0],
        )
    assert reconstructed_op == 2.0 * (operator - Operator({PAULI_IDENTITY: 2}))

    # Trotter order 2
    trotter_unbound_circuit = get_trotter_time_evolution_operator(
        operator, n_state_qubits=2, n_trotter=1, trotter_order=2
    )
    trotter_circuit = trotter_unbound_circuit.bind_parameters([2.0])
    reconstructed_op = Operator({})
    for g in trotter_circuit.gates:
        reconstructed_op.add_term(
            PauliLabel.from_index_and_pauli_list(g.target_indices, g.pauli_ids),
            0.5 * g.params[0],
        )
    assert reconstructed_op == 2 * (operator - Operator({PAULI_IDENTITY: 2}))


def test_trotter_factory() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
    )
    problem = QubitHamiltonian(2, operator)
    circuit_factory = FixedStepTrotterTimeEvolutionCircuitFactory(problem, 1)
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(2)
    expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 4.0)
    expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 4.0)
    assert circuit == expected_circuit

    n_trotter = 5
    circuit_factory = FixedStepTrotterTimeEvolutionCircuitFactory(problem, n_trotter)
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(2)
    for _ in range(n_trotter):
        expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 4.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 4.0 / n_trotter)
    assert circuit == expected_circuit

    trotter_order = 2
    n_trotter = 3
    circuit_factory = FixedStepTrotterTimeEvolutionCircuitFactory(
        problem, n_trotter, trotter_order
    )
    circuit = circuit_factory(1.0)

    expected_circuit = QuantumCircuit(2)
    for _ in range(n_trotter):
        expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2.0 / n_trotter)

    assert circuit == expected_circuit


def test_controlled_trotter_factory() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
    )
    problem = QubitHamiltonian(2, operator)
    circuit_factory = FixedStepTrotterControlledTimeEvolutionCircuitFactory(problem, 1)
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(3)
    expected_circuit.add_PauliRotation_gate([1, 2], [1, 1], 2.0)
    expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -2.0)
    expected_circuit.add_PauliRotation_gate([1, 2], [2, 2], 2.0)
    expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -2.0)
    expected_circuit.add_RZ_gate(0, -2.0)
    assert circuit == expected_circuit

    # Trotter steps 5
    n_trotter = 5
    circuit_factory = FixedStepTrotterControlledTimeEvolutionCircuitFactory(
        problem, n_trotter
    )
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(3)
    for _ in range(n_trotter):
        expected_circuit.add_PauliRotation_gate([1, 2], [1, 1], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([1, 2], [2, 2], 2.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -2.0 / n_trotter)
        expected_circuit.add_RZ_gate(0, -2.0 / n_trotter)
    assert circuit == expected_circuit

    # trotter order 2
    n_trotter = 5
    circuit_factory = FixedStepTrotterControlledTimeEvolutionCircuitFactory(
        problem, n_trotter, trotter_order=2
    )
    circuit = circuit_factory(1.0)
    expected_circuit = QuantumCircuit(3)
    for _ in range(n_trotter):
        expected_circuit.add_PauliRotation_gate([1, 2], [1, 1], 1.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -1.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([1, 2], [2, 2], 1.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -1.0 / n_trotter)
        expected_circuit.add_RZ_gate(0, -1.0 / n_trotter)
        expected_circuit.add_RZ_gate(0, -1.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([1, 2], [2, 2], 1.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -1.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([1, 2], [1, 1], 1.0 / n_trotter)
        expected_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -1.0 / n_trotter)

    assert circuit == expected_circuit


def test_fixed_interval_trotter() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
    )
    problem = QubitHamiltonian(2, operator)
    time_step = 0.1
    circuit_factory = FixedIntervalTrotterTimeEvolutionCircuitFactory(
        problem, time_step, trotter_order=2
    )

    single_step_circuit = QuantumCircuit(2)
    single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2 * time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2 * time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2 * time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2 * time_step)

    # Positive evolution time
    evolution_time = 0.5
    circuit = circuit_factory(evolution_time)
    expected = QuantumCircuit(2)
    for _ in range(int(evolution_time / time_step)):
        expected.extend(single_step_circuit)

    assert expected == circuit

    # Negative evolution time
    expected = QuantumCircuit(2)
    evolution_time = -0.5
    circuit = circuit_factory(evolution_time)
    for _ in range(int(np.abs(evolution_time) / time_step)):
        expected.extend(single_step_circuit)
    expected = inverse_circuit(expected)

    assert expected == circuit

    # Erroneous evolution time
    evolution_time = 0.43
    with pytest.raises(
        ValueError,
        match=f"Evolution time {evolution_time} is not an integer multiple of time step {time_step}.",
    ):
        circuit_factory(evolution_time)


def test_fixed_interval_controlled_trotter() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
    )
    problem = QubitHamiltonian(2, operator)
    time_step = 0.1
    circuit_factory = FixedIntervalTrotterControlledTimeEvolutionCircuitFactory(
        problem, time_step, trotter_order=2
    )

    single_step_circuit = QuantumCircuit(3)
    single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -time_step)
    single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -time_step)
    single_step_circuit.add_RZ_gate(0, -time_step)
    single_step_circuit.add_RZ_gate(0, -time_step)
    single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -time_step)
    single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], time_step)
    single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -time_step)

    # Positive evolution time
    evolution_time = 0.5
    circuit = circuit_factory(evolution_time)
    expected = QuantumCircuit(3)
    for _ in range(int(evolution_time / time_step)):
        expected.extend(single_step_circuit)

    assert expected == circuit

    # Negative evolution time
    expected = QuantumCircuit(3)
    evolution_time = -0.5
    circuit = circuit_factory(evolution_time)
    for _ in range(int(np.abs(evolution_time) / time_step)):
        expected.extend(single_step_circuit)
    expected = inverse_circuit(expected)

    assert expected == circuit

    # Erroneous evolution time
    evolution_time = 0.43
    with pytest.raises(
        ValueError,
        match=f"Evolution time {evolution_time} is not an integer multiple of time step {time_step}.",
    ):
        circuit_factory(evolution_time)


class TestTrotterTimeEvolutionCircuitFactory(unittest.TestCase):
    operator: Operator
    problem: QubitHamiltonian

    @classmethod
    def setUpClass(cls) -> None:
        cls.operator = Operator(
            {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
        )
        cls.problem = QubitHamiltonian(2, cls.operator)

    def test_fixed_trotter_step(self) -> None:
        evolution_time = 0.5
        n_trotter = 3
        factory = TrotterTimeEvolutionCircuitFactory(self.problem, n_trotter=n_trotter)

        single_step_circuit = QuantumCircuit(2)
        single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2 / n_trotter)

        expected_circuit = QuantumCircuit(2)
        for _ in range(n_trotter):
            expected_circuit.extend(single_step_circuit)

        assert expected_circuit == factory(evolution_time)

    def test_fixed_time_step(self) -> None:
        time_step = 0.1
        evolution_time = 0.5

        factory = TrotterTimeEvolutionCircuitFactory(self.problem, time_step=time_step)

        single_step_circuit = QuantumCircuit(2)
        single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 4 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 4 * time_step)

        expected_circuit = QuantumCircuit(2)
        for _ in range(int(evolution_time / time_step)):
            expected_circuit.extend(single_step_circuit)

        assert expected_circuit == factory(evolution_time)

    def test_fixed_trotter_step_2nd_order(self) -> None:
        evolution_time = 0.5
        n_trotter = 3
        factory = TrotterTimeEvolutionCircuitFactory(
            self.problem, n_trotter=n_trotter, trotter_order=2
        )

        single_step_circuit = QuantumCircuit(2)

        single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 1 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 1 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 1 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 1 / n_trotter)

        expected_circuit = QuantumCircuit(2)
        for _ in range(n_trotter):
            expected_circuit.extend(single_step_circuit)

        assert expected_circuit == factory(evolution_time)

    def test_fixed_time_step_2nd_order(self) -> None:
        time_step = 0.1
        evolution_time = 0.5

        factory = TrotterTimeEvolutionCircuitFactory(
            self.problem, time_step=time_step, trotter_order=2
        )

        single_step_circuit = QuantumCircuit(2)
        single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1], [2, 2], 2 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1], [1, 1], 2 * time_step)

        expected_circuit = QuantumCircuit(2)
        for _ in range(int(evolution_time / time_step)):
            expected_circuit.extend(single_step_circuit)

        assert expected_circuit == factory(evolution_time)


class TestTrotterControlledTimeEvolutionCircuitFactory(unittest.TestCase):
    operator: Operator
    problem: QubitHamiltonian

    @classmethod
    def setUpClass(cls) -> None:
        cls.operator = Operator(
            {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, PAULI_IDENTITY: 2}
        )
        cls.problem = QubitHamiltonian(2, cls.operator)

    def test_fixed_trotter_step(self) -> None:
        evolution_time = 0.5
        n_trotter = 3
        factory = TrotterControlledTimeEvolutionCircuitFactory(
            self.problem, n_trotter=n_trotter
        )

        single_step_circuit = QuantumCircuit(3)
        single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], 1 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -1 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], 1 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -1 / n_trotter)
        single_step_circuit.add_RZ_gate(0, -1 / n_trotter)

        expected_circuit = QuantumCircuit(3)
        for _ in range(n_trotter):
            expected_circuit.extend(single_step_circuit)
        assert expected_circuit == factory(evolution_time)

    def test_fixed_time_step(self) -> None:
        time_step = 0.1
        evolution_time = 0.5

        factory = TrotterControlledTimeEvolutionCircuitFactory(
            self.problem, time_step=time_step
        )

        single_step_circuit = QuantumCircuit(3)
        single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], 2 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -2 * time_step)
        single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], 2 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -2 * time_step)
        single_step_circuit.add_RZ_gate(0, -2 * time_step)

        expected_circuit = QuantumCircuit(3)
        for _ in range(int(evolution_time / time_step)):
            expected_circuit.extend(single_step_circuit)

        assert expected_circuit == factory(evolution_time)

    def test_fixed_trotter_step_2nd_order(self) -> None:
        evolution_time = 0.5
        n_trotter = 3
        factory = TrotterControlledTimeEvolutionCircuitFactory(
            self.problem, n_trotter=n_trotter, trotter_order=2
        )

        single_step_circuit = QuantumCircuit(3)
        single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], 0.5 / n_trotter)
        single_step_circuit.add_PauliRotation_gate(
            [0, 1, 2], [3, 1, 1], -0.5 / n_trotter
        )
        single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], 0.5 / n_trotter)
        single_step_circuit.add_PauliRotation_gate(
            [0, 1, 2], [3, 2, 2], -0.5 / n_trotter
        )
        single_step_circuit.add_RZ_gate(0, -0.5 / n_trotter)
        single_step_circuit.add_RZ_gate(0, -0.5 / n_trotter)
        single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], 0.5 / n_trotter)
        single_step_circuit.add_PauliRotation_gate(
            [0, 1, 2], [3, 2, 2], -0.5 / n_trotter
        )
        single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], 0.5 / n_trotter)
        single_step_circuit.add_PauliRotation_gate(
            [0, 1, 2], [3, 1, 1], -0.5 / n_trotter
        )

        expected_circuit = QuantumCircuit(3)
        for _ in range(n_trotter):
            expected_circuit.extend(single_step_circuit)
        assert expected_circuit == factory(evolution_time)

    def test_fixed_time_step_2nd_order(self) -> None:
        time_step = 0.1
        evolution_time = 0.5

        factory = TrotterControlledTimeEvolutionCircuitFactory(
            self.problem, time_step=time_step, trotter_order=2
        )

        single_step_circuit = QuantumCircuit(3)
        single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], 1 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -1 * time_step)
        single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], 1 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -1 * time_step)
        single_step_circuit.add_RZ_gate(0, -1 * time_step)
        single_step_circuit.add_RZ_gate(0, -1 * time_step)
        single_step_circuit.add_PauliRotation_gate([1, 2], [2, 2], 1 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 2, 2], -1 * time_step)
        single_step_circuit.add_PauliRotation_gate([1, 2], [1, 1], 1 * time_step)
        single_step_circuit.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -1 * time_step)

        expected_circuit = QuantumCircuit(3)
        for _ in range(int(evolution_time / time_step)):
            expected_circuit.extend(single_step_circuit)

        assert expected_circuit == factory(evolution_time)
