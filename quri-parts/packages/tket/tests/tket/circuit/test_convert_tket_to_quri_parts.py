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
from pytket.circuit import Circuit, OpType, Unitary1qBox, Unitary2qBox, Unitary3qBox
from scipy.stats import unitary_group

from quri_parts.circuit import QuantumGate, gates
from quri_parts.tket.circuit.tket_circuit_converter import circuit_from_tket


def gate_equal(gate_1: QuantumGate, gate_2: QuantumGate) -> None:
    assert gate_1.name == gate_2.name
    assert gate_1.control_indices == gate_2.control_indices
    assert gate_1.target_indices == gate_2.target_indices
    assert np.allclose(
        (np.array(gate_1.params) / np.pi) % 2, (np.array(gate_2.params) / np.pi) % 2
    )


def test_circuit_from_tket() -> None:
    unitary_1q_matrix = [[1, 0], [0, np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4)]]
    unitary_2q_matrix = unitary_group.rvs(4)
    unitary_3q_matrix = unitary_group.rvs(8)

    tket_circuit = Circuit(4)
    tket_circuit.add_gate(OpType.noop, (0,))
    tket_circuit.X(1)
    tket_circuit.Y(2)
    tket_circuit.Z(3)
    tket_circuit.H(0)
    tket_circuit.S(1)
    tket_circuit.Sdg(2)
    tket_circuit.SX(3)
    tket_circuit.SXdg(0)
    tket_circuit.T(1)
    tket_circuit.Tdg(2)
    tket_circuit.Rx(0.387, 3)
    tket_circuit.Ry(-9.956, 0)
    tket_circuit.Rz(3.643, 1)
    tket_circuit.U1(0.387, 2)
    tket_circuit.U2(0.387, -9.956, 3)
    tket_circuit.U3(0.387, -9.956, 3.643, 0)
    tket_circuit.CX(0, 1)
    tket_circuit.CZ(1, 2)
    tket_circuit.SWAP(0, 2)
    tket_circuit.CCX(0, 1, 2)
    tket_circuit.add_unitary1qbox(Unitary1qBox(np.array(unitary_1q_matrix)), 2)
    tket_circuit.add_unitary2qbox(Unitary2qBox(unitary_2q_matrix), 0, 2)
    tket_circuit.add_unitary3qbox(Unitary3qBox(unitary_3q_matrix), 0, 2, 3)

    qp_circuit = circuit_from_tket(tket_circuit)

    gate_sequence = [
        gates.Identity(0),
        gates.X(1),
        gates.Y(2),
        gates.Z(3),
        gates.H(0),
        gates.S(1),
        gates.Sdag(2),
        gates.SqrtX(3),
        gates.SqrtXdag(0),
        gates.T(1),
        gates.Tdag(2),
        gates.RX(3, 0.387 * np.pi),
        gates.RY(0, -9.956 * np.pi),
        gates.RZ(1, 3.643 * np.pi),
        gates.U1(2, (0.387) * np.pi),
        gates.U2(3, (0.387) * np.pi, -9.956 * np.pi),
        gates.U3(0, (0.387) * np.pi, -9.956 * np.pi, (3.643) * np.pi),
        gates.CNOT(0, 1),
        gates.CZ(1, 2),
        gates.SWAP(0, 2),
        gates.TOFFOLI(0, 1, 2),
        gates.UnitaryMatrix([2], unitary_1q_matrix),
        gates.UnitaryMatrix([0, 2], unitary_2q_matrix),
        gates.UnitaryMatrix([0, 2, 3], unitary_3q_matrix),
    ]

    for i, g in enumerate(qp_circuit.gates):
        gate_equal(gate_sequence[i], g)
