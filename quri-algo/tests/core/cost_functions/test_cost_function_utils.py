# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pytest import raises
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label

from quri_algo.core.cost_functions.utils import (
    complex_conjugate_circuit,
    expand_circuit,
    get_hs_operator,
    prepare_circuit_hilbert_schmidt_test,
)


def test_complex_conjugate_circuit() -> None:
    qubit_count = 4

    circuit = QuantumCircuit(qubit_count)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_CNOT_gate(1, 2)
    circuit.add_CNOT_gate(2, 3)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    circuit.add_H_gate(2)
    circuit.add_H_gate(3)
    circuit.add_RX_gate(0, 0.5)
    circuit.add_RY_gate(1, 0.5)
    circuit.add_PauliRotation_gate(
        tuple(i for i in range(qubit_count - 1)),
        tuple(2 for _ in range(qubit_count - 1)),
        0.5,
    )
    circuit.add_PauliRotation_gate(
        tuple(i for i in range(qubit_count)), tuple(2 for _ in range(qubit_count)), 0.5
    )

    circuit_conjugated_expected = QuantumCircuit(qubit_count)
    circuit_conjugated_expected.add_CNOT_gate(0, 1)
    circuit_conjugated_expected.add_CNOT_gate(1, 2)
    circuit_conjugated_expected.add_CNOT_gate(2, 3)
    circuit_conjugated_expected.add_H_gate(0)
    circuit_conjugated_expected.add_H_gate(1)
    circuit_conjugated_expected.add_H_gate(2)
    circuit_conjugated_expected.add_H_gate(3)
    circuit_conjugated_expected.add_RX_gate(0, -0.5)
    circuit_conjugated_expected.add_RY_gate(1, 0.5)
    circuit_conjugated_expected.add_PauliRotation_gate(
        tuple(i for i in range(qubit_count - 1)),
        tuple(2 for _ in range(qubit_count - 1)),
        0.5,
    )
    circuit_conjugated_expected.add_PauliRotation_gate(
        tuple(i for i in range(qubit_count)), tuple(2 for _ in range(qubit_count)), -0.5
    )

    circuit_conjugated = complex_conjugate_circuit(circuit)

    assert circuit_conjugated == circuit_conjugated_expected


def test_expand_circuit() -> None:
    qubit_count = 4
    shift = 2
    new_qubit_count = 8

    circuit = QuantumCircuit(qubit_count)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_CNOT_gate(1, 2)
    circuit.add_CNOT_gate(2, 3)
    circuit.add_H_gate(0)
    circuit.add_H_gate(1)
    circuit.add_H_gate(2)
    circuit.add_H_gate(3)
    circuit.add_RX_gate(0, 0.5)
    circuit.add_RX_gate(1, 0.5)

    circuit_expanded_expected = QuantumCircuit(new_qubit_count)
    circuit_expanded_expected.add_CNOT_gate(2, 3)
    circuit_expanded_expected.add_CNOT_gate(3, 4)
    circuit_expanded_expected.add_CNOT_gate(4, 5)
    circuit_expanded_expected.add_H_gate(2)
    circuit_expanded_expected.add_H_gate(3)
    circuit_expanded_expected.add_H_gate(4)
    circuit_expanded_expected.add_H_gate(5)
    circuit_expanded_expected.add_RX_gate(2, 0.5)
    circuit_expanded_expected.add_RX_gate(3, 0.5)

    circuit_expanded = expand_circuit(circuit, shift, new_qubit_count)
    assert circuit_expanded_expected == circuit_expanded


def test_get_hs_operator() -> None:
    lattice_size = 2
    alpha = 0.7

    global_hst = Operator(
        {
            PAULI_IDENTITY: 15 / 16,
            pauli_label("X 0 X 2"): -1 / 16,
            pauli_label("Y 0 Y 2"): 1 / 16,
            pauli_label("Z 0 Z 2"): -1 / 16,
            pauli_label("X 1 X 3"): -1 / 16,
            pauli_label("Y 1 Y 3"): 1 / 16,
            pauli_label("Z 1 Z 3"): -1 / 16,
            pauli_label("X 0 X 1 X 2 X 3"): -1 / 16,
            pauli_label("X 0 Y 1 X 2 Y 3"): 1 / 16,
            pauli_label("X 0 Z 1 X 2 Z 3"): -1 / 16,
            pauli_label("Y 0 X 1 Y 2 X 3"): 1 / 16,
            pauli_label("Y 0 Y 1 Y 2 Y 3"): -1 / 16,
            pauli_label("Y 0 Z 1 Y 2 Z 3"): 1 / 16,
            pauli_label("Z 0 X 1 Z 2 X 3"): -1 / 16,
            pauli_label("Z 0 Y 1 Z 2 Y 3"): 1 / 16,
            pauli_label("Z 0 Z 1 Z 2 Z 3"): -1 / 16,
        },
    )
    local_hst = Operator(
        {
            PAULI_IDENTITY: 3 / 4,
            pauli_label("X 0 X 2"): -1 / 8,
            pauli_label("Y 0 Y 2"): 1 / 8,
            pauli_label("Z 0 Z 2"): -1 / 8,
            pauli_label("X 1 X 3"): -1 / 8,
            pauli_label("Y 1 Y 3"): 1 / 8,
            pauli_label("Z 1 Z 3"): -1 / 8,
        },
    )
    expected_combined_hst = alpha * global_hst + (1 - alpha) * local_hst

    expected_circuits = [global_hst, local_hst, expected_combined_hst]
    actual_circuits = [get_hs_operator(a, lattice_size) for a in [1.0, 0.0, alpha]]

    with raises(ValueError):
        get_hs_operator(-0.1, lattice_size)
    with raises(ValueError):
        get_hs_operator(1.1, lattice_size)
    for e, a in zip(expected_circuits, actual_circuits):
        assert e == a


def test_prepare_circuit_hilbert_schmidt_test() -> None:
    angle = 0.5
    lattice_size = 2
    target_circuit = QuantumCircuit(lattice_size)
    target_circuit.add_H_gate(0)
    target_circuit.add_CNOT_gate(0, 1)
    trial_circuit = QuantumCircuit(lattice_size)
    trial_circuit.add_H_gate(0)
    trial_circuit.add_CNOT_gate(0, 1)
    trial_circuit.add_RZ_gate(0, angle)

    expected_circuit = QuantumCircuit(2 * lattice_size)
    for i in range(lattice_size):
        expected_circuit.add_H_gate(i)
        expected_circuit.add_CNOT_gate(i, lattice_size + i)
    expected_circuit.add_H_gate(lattice_size)
    expected_circuit.add_CNOT_gate(lattice_size, lattice_size + 1)
    expected_circuit.add_RZ_gate(lattice_size, -angle)
    expected_circuit.add_H_gate(0)
    expected_circuit.add_CNOT_gate(0, 1)

    actual_circuit = prepare_circuit_hilbert_schmidt_test(
        target_circuit, trial_circuit
    ).circuit

    assert actual_circuit == expected_circuit.freeze()
