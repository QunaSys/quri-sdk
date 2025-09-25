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
from quri_parts.circuit import RX, NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    get_sparse_matrix,
    pauli_label,
)
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.sampler import create_qulacs_vector_ideal_sampler
from quri_parts.qulacs.simulator import evaluate_state_to_vector
from scipy.linalg import expm

from quri_algo.circuit.hadamard_test import construct_hadamard_circuit
from quri_algo.core.estimator import State
from quri_algo.core.estimator.time_evolution.trotter import (
    TrotterTimeEvolutionHadamardTest,
)
from quri_algo.problem import QubitHamiltonian


class TestTrotterTimeEvoHadamardTest(unittest.TestCase):
    problem: QubitHamiltonian
    estimator: TrotterTimeEvolutionHadamardTest[State]

    @classmethod
    def setUpClass(cls) -> None:
        operator = Operator(
            {
                pauli_label("X0 X1"): 2,
                pauli_label("Y0 Y1"): 2,
                pauli_label("Z0 Z1"): 2,
                PAULI_IDENTITY: 2,
            }
        )
        cls.problem = QubitHamiltonian(2, operator)
        cls.estimator = TrotterTimeEvolutionHadamardTest(
            qubit_hamiltonian=cls.problem,
            sampler=create_qulacs_vector_ideal_sampler(),
            n_trotter=1,
        )

    def test_trotter_hadamard_test(self) -> None:
        circuit = QuantumCircuit(
            2, gates=[RX(0, np.random.random()), RX(1, np.random.random())]
        )
        state = quantum_state(2, bits=0b01, circuit=circuit)
        evolution_time = 10.0
        hadamard_test_result = self.estimator(state, evolution_time, 10000)

        hamiltonian_matrix = get_sparse_matrix(self.problem.qubit_hamiltonian).toarray()
        exact_time_evo = expm(-1j * evolution_time * hamiltonian_matrix)
        state_vector = evaluate_state_to_vector(state).vector
        expected_result = state_vector.conj() @ exact_time_evo @ state_vector
        assert np.isclose(hadamard_test_result.value, expected_result)

    def test_circuit_factory(self) -> None:
        evo_time = 1.0
        circuit = self.estimator.real_circuit_factory(evo_time)
        expected_circuit = construct_hadamard_circuit(
            self.estimator.controlled_time_evolution_factory(evo_time), test_real=True
        )
        assert circuit == expected_circuit

        circuit = self.estimator.imag_circuit_factory(evo_time)
        expected_circuit = construct_hadamard_circuit(
            self.estimator.controlled_time_evolution_factory(evo_time), test_real=False
        )
        assert circuit == expected_circuit


def fake_transpiler(
    circuit: NonParametricQuantumCircuit,
) -> NonParametricQuantumCircuit:
    return QuantumCircuit(circuit.qubit_count)


def test_trotter_hadamard_test_with_transpiler() -> None:
    operator = Operator(
        {pauli_label("X0 X1"): 2, pauli_label("Y0 Y1"): 2, pauli_label("Z0 Z1"): 2}
    )
    problem = QubitHamiltonian(2, operator)
    estimator: TrotterTimeEvolutionHadamardTest[
        State
    ] = TrotterTimeEvolutionHadamardTest(
        qubit_hamiltonian=problem,
        sampler=create_qulacs_vector_ideal_sampler(),
        n_trotter=1,
        transpiler=fake_transpiler,
    )
    assert estimator.controlled_time_evolution_factory(
        np.random.random()
    ) == QuantumCircuit(3)

    circuit = QuantumCircuit(
        2, gates=[RX(0, np.random.random()), RX(1, np.random.random())]
    )
    state = quantum_state(2, bits=0b01, circuit=circuit)
    evolution_time = 10.0
    hadamard_test_result = estimator(state, evolution_time, 10000)

    expected_result = 1 + 1j
    assert np.isclose(hadamard_test_result.value, expected_result)
