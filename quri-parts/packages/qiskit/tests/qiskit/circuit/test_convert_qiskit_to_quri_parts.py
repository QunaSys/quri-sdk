# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from quri_parts.circuit import QuantumCircuit, gate_names, gates
from quri_parts.circuit.gate_names import is_unitary_matrix_gate_name
from quri_parts.qiskit.circuit import circuit_from_qiskit
from quri_parts.qulacs.circuit import convert_circuit


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


def test_circuit_from_qiskit_with_ecr() -> None:
    qis_circ = QiskitCircuit(2)
    qis_circ.ecr(0, 1)

    qp_circuit = circuit_from_qiskit(qis_circ)
    qulacs_circuit = convert_circuit(qp_circuit)

    # ECR should be decomposed into native gates instead of a raw unitary matrix.
    unitary_gate_names = [
        gate.name for gate in qp_circuit.gates if is_unitary_matrix_gate_name(gate.name)
    ]
    assert unitary_gate_names.count(gate_names.UnitaryMatrix) == 0

    expected = Operator(qis_circ).data
    actual = _qulacs_circuit_to_matrix(qulacs_circuit)
    _assert_unitary_equal_up_to_global_phase(actual, expected)


def _qulacs_circuit_to_matrix(circuit: Any) -> NDArray[np.complex128]:
    qubit_count = circuit.get_qubit_count()
    dim = 1 << qubit_count
    mat: NDArray[np.complex128] = np.zeros((dim, dim), dtype=np.complex128)
    from qulacs import QuantumState

    state = QuantumState(qubit_count)
    for col in range(dim):
        state.set_computational_basis(col)
        circuit.update_quantum_state(state)
        mat[:, col] = state.get_vector()
    return mat


def _assert_unitary_equal_up_to_global_phase(
    actual: NDArray[np.complex128],
    expected: NDArray[np.complex128],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> None:
    idx = np.argmax(np.abs(expected))
    assert np.abs(expected.flat[idx]) > 0
    phase = expected.flat[idx] / actual.flat[idx]
    assert np.allclose(actual * phase, expected, atol=atol, rtol=rtol)
