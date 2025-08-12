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

from numpy import allclose, angle, array, complex128, exp
from numpy.typing import NDArray

from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit
from quri_parts.core.state import GeneralCircuitQuantumState, QuantumStateVector
from quri_parts.stim.simulator import evaluate_state_to_vector, run_circuit


class TestSimulator(unittest.TestCase):
    update_circuit: ImmutableQuantumCircuit
    expected_output: NDArray[complex128]

    @classmethod
    def setUpClass(cls) -> None:
        n = 2
        cls.update_circuit = QuantumCircuit(n)
        cls.update_circuit.add_H_gate(0)
        cls.update_circuit.add_H_gate(1)
        cls.update_circuit.add_X_gate(0)
        cls.update_circuit.add_Y_gate(1)
        cls.update_circuit.add_SqrtY_gate(0)
        cls.update_circuit.add_SqrtX_gate(1)

        cls.expected_output = array([0.0, 0.0, -0.5 - 0.5j, -0.5 - 0.5j])
        # This is to cope with the fact that stim always canonicalize the
        # state vector such that the first non-zero component is positive.
        phase = exp(1j * angle(cls.expected_output[cls.expected_output != 0][0]))
        cls.expected_output = cls.expected_output / phase

    def test_evaluate_state_to_vector_general_circuit_state_to_vec(self) -> None:
        circuit_quantum_state = GeneralCircuitQuantumState(2, self.update_circuit)
        quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)
        assert len(quantum_state_vector.circuit.gates) == 0
        assert allclose(quantum_state_vector.vector, [0, 0.70710678, 0, -0.70710678])

    def test_evaluate_state_to_vector_vec_to_vec(self) -> None:
        circuit_quantum_state = QuantumStateVector(
            2, array([0.5, -0.5, 0.5j, -0.5j]), self.update_circuit
        )
        quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)

        assert len(quantum_state_vector.circuit.gates) == 0
        assert allclose(quantum_state_vector.vector, self.expected_output)

    def test_run_circuit_simple(self) -> None:
        updated_quantum_state_vector = run_circuit(
            self.update_circuit, array([0.5, -0.5, 0.5j, -0.5j])
        )
        assert allclose(updated_quantum_state_vector, self.expected_output)
