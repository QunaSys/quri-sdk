# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensornetwork as tn
from numpy.testing import assert_almost_equal

from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.gates import CNOT, TOFFOLI, H, S, Y
from quri_parts.core.state import ComputationalBasisState, GeneralCircuitQuantumState
from quri_parts.tensornetwork.circuit import convert_circuit
from quri_parts.tensornetwork.state import convert_state

circuit_tensor_pairs = [
    (
        QuantumCircuit(2, gates=[H(0), CNOT(0, 1)]),
        np.array([[1 / np.sqrt(2), 0.0], [0.0, 1 / np.sqrt(2)]]),
    ),
    (
        QuantumCircuit(2, gates=[H(0)]),
        np.array([[1 / np.sqrt(2), 0.0], [1 / np.sqrt(2), 0.0]]),
    ),
    (
        QuantumCircuit(2, gates=[H(0), H(1), S(1)]),
        np.array([[1 / 2, 1j / 2], [1 / 2, 1j / 2]]),
    ),
    (
        QuantumCircuit(2, gates=[H(0), CNOT(0, 1), Y(0)]),
        np.array([[0.0, -1j / np.sqrt(2)], [1j / np.sqrt(2), 0.0]]),
    ),
    (
        QuantumCircuit(2, gates=[H(0), CNOT(0, 1), Y(1)]),
        np.array([[0.0, 1j / np.sqrt(2)], [-1j / np.sqrt(2), 0.0]]),
    ),
    (
        QuantumCircuit(3, gates=[H(0), CNOT(0, 1), TOFFOLI(0, 1, 2)]),
        np.array(
            [[[1 / np.sqrt(2), 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1 / np.sqrt(2)]]]
        ),
    ),
    (
        QuantumCircuit(3, gates=[H(0), CNOT(0, 1), CNOT(0, 2)]),
        np.array(
            [[[1 / np.sqrt(2), 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1 / np.sqrt(2)]]]
        ),
    ),
]


def test_convert_state() -> None:
    for c, t in circuit_tensor_pairs:
        state = GeneralCircuitQuantumState(c.qubit_count, c)
        converted_state = convert_state(state)
        contracted_state = tn.contractors.optimal(
            converted_state._container, output_edge_order=converted_state.edges
        )
        assert_almost_equal(contracted_state.tensor, t)


def test_conjugate_state() -> None:
    for c, t in circuit_tensor_pairs:
        state = GeneralCircuitQuantumState(c.qubit_count, c)
        converted_state = convert_state(state)
        conjugated_state = converted_state.conjugate()
        contracted_state = tn.contractors.optimal(
            conjugated_state._container, output_edge_order=conjugated_state.edges
        )
        assert_almost_equal(contracted_state.tensor, np.conj(t))


def test_with_gates_applied() -> None:
    for c, t in circuit_tensor_pairs:
        state = ComputationalBasisState(c.qubit_count, bits=0)
        converted_state = convert_state(state)
        tensornetwork_circuit = convert_circuit(c)
        state_with_gates_applied = converted_state.with_gates_applied(
            tensornetwork_circuit
        )
        contracted_state = tn.contractors.optimal(
            state_with_gates_applied._container,
            output_edge_order=state_with_gates_applied.edges,
        )
        assert_almost_equal(contracted_state.tensor, t)
