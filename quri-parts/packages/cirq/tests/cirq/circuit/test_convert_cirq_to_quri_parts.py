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
from cirq.circuits.circuit import Circuit as CirqCircuit
from cirq.devices.line_qubit import LineQubit
from cirq.ops.common_gates import CNOT, CZ, H, Rx, Rz, T
from cirq.ops.matrix_gates import MatrixGate
from cirq.ops.pauli_gates import X
from cirq.ops.swap_gates import SWAP
from cirq.ops.three_qubit_gates import TOFFOLI

from quri_parts.circuit import QuantumCircuit, gates
from quri_parts.cirq.circuit.cirq_circuit_converter import circuit_from_cirq


def test_circuit_from_cirq() -> None:
    circuit = CirqCircuit()
    q0 = LineQubit(0)
    q1 = LineQubit(1)
    q2 = LineQubit(2)
    mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    matrix_gate = MatrixGate(mat)

    circuit.append(
        [
            H(q0),
            T(q1),
            X(q2) ** -1,
            CZ(q0, q1),
            H(q2) ** 3,
            CNOT(q1, q2),
            SWAP(q0, q1),
            TOFFOLI(q0, q1, q2),
            Rx(rads=0.5).on(q1),
            Rz(rads=1.25).on(q0),
            matrix_gate(q0, q1),
        ]
    )

    gate_list = [
        gates.H(0),
        gates.T(1),
        gates.X(2),
        gates.CZ(0, 1),
        gates.H(2),
        gates.CNOT(1, 2),
        gates.SWAP(0, 1),
        gates.TOFFOLI(0, 1, 2),
        gates.RX(1, 0.5),
        gates.RZ(0, 1.25),
        gates.UnitaryMatrix([0, 1], mat.tolist()),
    ]
    expected = QuantumCircuit(3, gates=gate_list)

    assert expected.gates == circuit_from_cirq(circuit).gates
