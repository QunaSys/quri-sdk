# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from quri_parts.circuit import (
    H,
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
    QuantumCircuit,
    X,
)
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.circuit.utils.controlled_rotations import add_controlled_RY_gate
from quri_parts.core.state import CircuitQuantumState, quantum_state
from quri_parts.qulacs.sampler import create_qulacs_vector_ideal_sampler
from scipy.linalg import expm

from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.estimator.hadamard_test import (
    HadamardTest,
    get_hadamard_test_ancilla_qubit_counter,
    remap_state_for_hadamard_test,
    shift_state_circuit,
)
from quri_algo.problem import Problem


class FakeProblem(Problem):
    def __init__(self, n_state_qubit: int):
        self.n_state_qubit = n_state_qubit


class ControlledRYFactory(CircuitFactory):
    def __init__(
        self,
        encoded_problem: FakeProblem,
        *,
        transpiler: CircuitTranspiler | None = None
    ):
        self.encoded_problem = encoded_problem
        self.qubit_count = self.encoded_problem.n_state_qubit + 1
        self.transpiler = transpiler
        self._param_circuit = LinearMappedUnboundParametricQuantumCircuit(
            self.qubit_count
        )
        p = self._param_circuit.add_parameter("p")
        add_controlled_RY_gate(self._param_circuit, 0, 1, {p: -2})

    def __call__(self, angle: float) -> NonParametricQuantumCircuit:
        return self._param_circuit.bind_parameters([angle])


def test_hadamard_test() -> None:
    problem = FakeProblem(1)
    sampler = create_qulacs_vector_ideal_sampler()
    circuit_generator = ControlledRYFactory(problem)
    hadamard_test: HadamardTest[CircuitQuantumState] = HadamardTest(
        circuit_generator, sampler
    )
    circuit = QuantumCircuit(1, gates=[H(0)])
    state = quantum_state(1, circuit=circuit)

    theta = np.random.random()
    result = hadamard_test(state, 10000, theta)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    expected = expm(1j * theta * pauli_y).sum() / 2
    assert np.isclose(result.value, expected)


def test_get_hadamard_test_ancilla_qubit_counter() -> None:
    sampler_counts = {0: 10, 1: 1, 2: 20, 3: 2, 4: 30, 5: 3}
    counts_expected = {0: 60, 1: 6}
    counts = get_hadamard_test_ancilla_qubit_counter(sampler_counts)
    assert counts_expected == counts


def test_state_circuit() -> None:
    circuit = QuantumCircuit(2)
    circuit.add_X_gate(0)
    circuit.add_X_gate(1)

    expected_circuit = QuantumCircuit(3)
    expected_circuit.add_X_gate(1)
    expected_circuit.add_X_gate(2)

    shifted_circuit = shift_state_circuit(circuit)
    assert shifted_circuit == expected_circuit

    circuit = QuantumCircuit(2)
    circuit.add_X_gate(0)
    circuit.add_X_gate(1)

    expected_circuit = QuantumCircuit(8)
    expected_circuit.add_X_gate(6)
    expected_circuit.add_X_gate(7)

    shifted_circuit = shift_state_circuit(circuit, 6)
    assert shifted_circuit == expected_circuit


def test_remap_state_for_hadamard_test() -> None:
    state: CircuitQuantumState

    # test ComputationalBasisState
    state = quantum_state(3, bits=0b011)
    parsed_state = remap_state_for_hadamard_test(state)
    expected_state_circuit = QuantumCircuit(4, gates=[X(1), X(2)])
    assert parsed_state.circuit == expected_state_circuit

    # test GeneralCircuitQuantumState
    circuit = QuantumCircuit(3)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_RZ_gate(2, 1.0)
    state = quantum_state(3, bits=0b011, circuit=circuit)
    parsed_state = remap_state_for_hadamard_test(state)

    expected_state_circuit = QuantumCircuit(4, gates=[X(1), X(2)])
    expected_state_circuit.add_H_gate(1)
    expected_state_circuit.add_CNOT_gate(1, 2)
    expected_state_circuit.add_RZ_gate(3, 1.0)

    assert parsed_state.circuit == expected_state_circuit

    # test state vector
    n_qubits = 3
    statevector = np.random.rand(2**n_qubits)
    statevector /= np.linalg.norm(statevector)
    vector_state = quantum_state(n_qubits, vector=statevector)
    parsed_vector_state = remap_state_for_hadamard_test(vector_state)
    assert parsed_vector_state.qubit_count == 4
    assert np.allclose(parsed_vector_state.vector, np.kron(statevector, [1, 0]))

    # test state vector with circuit
    circuit = QuantumCircuit(3)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_RZ_gate(2, 1.0)

    statevector = np.random.rand(2**n_qubits)
    statevector /= np.linalg.norm(statevector)
    vector_state = quantum_state(n_qubits, vector=statevector, circuit=circuit)
    parsed_vector_state = remap_state_for_hadamard_test(vector_state)

    expected_state_circuit = QuantumCircuit(4)
    expected_state_circuit.add_H_gate(1)
    expected_state_circuit.add_CNOT_gate(1, 2)
    expected_state_circuit.add_RZ_gate(3, 1.0)

    assert parsed_vector_state.qubit_count == 4
    assert np.allclose(parsed_vector_state.vector, np.kron(statevector, [1, 0]))
    assert parsed_vector_state.circuit == expected_state_circuit
