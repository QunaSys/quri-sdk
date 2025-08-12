# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.circuit.library import UnitaryGate

from quri_parts.circuit import QuantumCircuit, gates
from quri_parts.qiskit.circuit import circuit_from_qiskit


def test_circuit_from_qiskit() -> None:
    qis_circ = QiskitCircuit(3)
    qis_circ.x(0)
    qis_circ.h(1)
    qis_circ.cx(0, 1)
    qis_circ.cz(0, 2)
    qis_circ.swap(1, 2)
    qis_circ.rx(0.125, 0)
    qis_circ.p(2.3, 0)
    qis_circ.u(1.2, 2.1, 3.1, 2)
    qis_circ.ccx(0, 1, 2)

    matrix = [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
    gate = UnitaryGate(matrix)
    qis_circ.append(gate, [0, 1])

    gate_list = [
        gates.X(0),
        gates.H(1),
        gates.CNOT(0, 1),
        gates.CZ(0, 2),
        gates.SWAP(1, 2),
        gates.RX(0, 0.125),
        gates.U1(0, 2.3),
        gates.U3(2, 1.2, 2.1, 3.1),
        gates.TOFFOLI(0, 1, 2),
        gates.UnitaryMatrix([0, 1], matrix),
    ]
    expected = QuantumCircuit(3, gates=gate_list)

    assert circuit_from_qiskit(qis_circ).gates == expected.gates
