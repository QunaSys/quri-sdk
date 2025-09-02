# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import Callable, Type, cast

import numpy as np
import numpy.typing as npt
import qiskit.circuit.library as qgate
import qiskit.quantum_info as qi
from qiskit import qasm3
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit.gate import Gate as QiskitGate
from qiskit.circuit.library import UnitaryGate

from quri_parts.circuit import QuantumCircuit, QuantumGate, gates
from quri_parts.circuit.transpile import TwoQubitUnitaryMatrixKAKTranspiler
from quri_parts.qiskit.circuit import convert_circuit, convert_gate


def gate_equal(i1: QiskitGate, i2: QiskitGate) -> bool:
    # Compare the matrix representation of the gates.
    return cast(bool, qi.Operator(i1).equiv(qi.Operator(i2)))


def circuit_equal(c1: QiskitQuantumCircuit, c2: QiskitQuantumCircuit) -> bool:
    # Compare the matrix representation of the circuits, including the global phase.
    return cast(bool, qi.Operator(c1).equiv(qi.Operator(c2)))


single_qubit_gate_mapping: Mapping[Callable[[int], QuantumGate], QiskitGate] = {
    gates.Identity: qgate.IGate(),
    gates.X: qgate.XGate(),
    gates.Y: qgate.YGate(),
    gates.Z: qgate.ZGate(),
    gates.H: qgate.HGate(),
    gates.S: qgate.SGate(),
    gates.Sdag: qgate.SdgGate(),
    gates.T: qgate.TGate(),
    gates.Tdag: qgate.TdgGate(),
    gates.SqrtX: qgate.SXGate(),
    gates.SqrtXdag: qgate.SXdgGate(),
}


def test_convert_single_qubit_gate() -> None:
    for qp_factory, qiskit_gate in single_qubit_gate_mapping.items():
        qp_gate = qp_factory(7)
        converted = convert_gate(qp_gate)
        expected = qiskit_gate
        assert gate_equal(converted, expected)


single_qubit_sgate_mapping: Mapping[Callable[[int], QuantumGate], QiskitGate] = {
    gates.SqrtY: UnitaryGate(
        np.array([[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]])
    ),
    gates.SqrtYdag: UnitaryGate(
        np.array([[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 - 0.5j]])
    ),
}


def test_convert_single_qubit_sgate() -> None:
    for qp_factory, qiskit_gate in single_qubit_sgate_mapping.items():
        qp_gate = qp_factory(7)
        converted = convert_gate(qp_gate)
        expected = qiskit_gate
        assert gate_equal(converted, expected)


rotation_gate_mapping: Mapping[
    Callable[[int, float], QuantumGate], Type[QiskitGate]
] = {
    gates.RX: qgate.RXGate,
    gates.RY: qgate.RYGate,
    gates.RZ: qgate.RZGate,
}


def test_convert_rotation_gate() -> None:
    for qp_factory, qiskit_type in rotation_gate_mapping.items():
        qp_gate = qp_factory(7, 0.125)
        converted = convert_gate(qp_gate)
        expected = qiskit_type(0.125)
        assert gate_equal(converted, expected)


two_qubit_gate_mapping: Mapping[Callable[[int, int], QuantumGate], QiskitGate] = {
    gates.CNOT: qgate.CXGate(),
    gates.CZ: qgate.CZGate(),
    gates.SWAP: qgate.SwapGate(),
}


def test_convert_two_qubit_gate() -> None:
    for qp_factory, qiskit_gate in two_qubit_gate_mapping.items():
        qp_gate = qp_factory(11, 7)
        converted = convert_gate(qp_gate)
        expected = qiskit_gate
        assert gate_equal(converted, expected)


three_qubit_gate_mapping: Mapping[
    Callable[[int, int, int], QuantumGate], QiskitGate
] = {
    gates.TOFFOLI: qgate.CCXGate(),
}


def test_convert_three_qubit_gate() -> None:
    for qp_factory, qiskit_gate in three_qubit_gate_mapping.items():
        qp_gate = qp_factory(11, 7, 5)
        converted = convert_gate(qp_gate)
        expected = qiskit_gate
        assert gate_equal(converted, expected)


def test_convert_unitary_matrix_gate() -> None:
    umat = ((1, 0), (0, np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4)))
    qp_gate = gates.UnitaryMatrix((7,), umat)
    converted = convert_gate(qp_gate)
    expected = UnitaryGate(np.array(umat))
    assert gate_equal(converted, expected)


def test_convert_u1_gate() -> None:
    lmd1 = 0.125
    converted = convert_gate(gates.U1(lmd=lmd1, target_index=0))
    expected = qgate.PhaseGate(lmd1)
    assert gate_equal(converted, expected)


def test_convert_u2_gate() -> None:
    phi2, lmd2 = 0.125, -0.125
    converted = convert_gate(gates.U2(phi=phi2, lmd=lmd2, target_index=0))
    expected = qgate.UGate(np.pi / 2, phi2, lmd2)
    assert gate_equal(converted, expected)


def test_convert_u3_gate() -> None:
    theta3, phi3, lmd3 = 0.125, -0.125, 0.625
    converted = convert_gate(gates.U3(theta=theta3, phi=phi3, lmd=lmd3, target_index=0))
    expected = qgate.UGate(theta3, phi3, lmd3)
    assert gate_equal(converted, expected)


def test_convert_circuit() -> None:
    circuit = QuantumCircuit(3)
    original_gates = [
        gates.X(1),
        gates.H(2),
        gates.CNOT(0, 2),
        gates.RX(0, 0.125),
    ]
    for gate in original_gates:
        circuit.add_gate(gate)

    converted = convert_circuit(circuit)
    assert converted.num_qubits == 3

    expected = QiskitQuantumCircuit(3)
    expected.x(1)
    expected.h(2)
    expected.cx(0, 2)
    expected.rx(0.125, 0)

    assert circuit_equal(converted, expected)


def test_convert_pauli() -> None:
    circuit = QuantumCircuit(5)
    original_gates = [
        gates.Pauli([0, 1, 3, 4], [1, 2, 3, 1]),
        gates.PauliRotation([0, 1, 3, 4], [1, 2, 3, 2], 0.266),
    ]
    for gate in original_gates:
        circuit.add_gate(gate)

    converted = convert_circuit(circuit, transpiler=None)
    assert converted.num_qubits == 5

    X = qi.SparsePauliOp("X")
    Y = qi.SparsePauliOp("Y")
    Z = qi.SparsePauliOp("Z")
    expected = QiskitQuantumCircuit(5)
    expected.pauli(pauli_string="XZYX", qubits=[0, 1, 3, 4])
    evo = qgate.PauliEvolutionGate(Y ^ Z ^ Y ^ X, time=0.133)
    expected.append(evo, [0, 1, 3, 4])

    assert circuit_equal(converted, expected)


def test_convert_kak_unitary_matrix() -> None:
    umat = [
        [
            -0.03177261 - 0.66874547j,
            -0.43306237 + 0.22683141j,
            -0.18247272 + 0.44450621j,
            0.21421016 - 0.18975362j,
        ],
        [
            -0.15150329 - 0.22323481j,
            0.21890311 - 0.42044443j,
            0.44248733 - 0.04540907j,
            0.70637772 + 0.07546113j,
        ],
        [
            -0.39591118 + 0.25563229j,
            -0.34085051 - 0.57044165j,
            -0.05519664 + 0.4884975j,
            -0.19727387 + 0.23607258j,
        ],
        [
            -0.37197857 - 0.34426935j,
            -0.30010596 + 0.06830857j,
            -0.01083026 - 0.57399229j,
            -0.14337561 + 0.54611345j,
        ],
    ]

    circuit = QuantumCircuit(4)
    circuit.add_TwoQubitUnitaryMatrix_gate(3, 1, umat)
    transpiled = TwoQubitUnitaryMatrixKAKTranspiler()(circuit)

    def get_unitary(circuit: QiskitQuantumCircuit) -> npt.NDArray[np.complex128]:
        sim = qi.Operator(circuit)
        return cast(
            npt.NDArray[np.complex128],
            sim.data,
        )

    target = get_unitary(convert_circuit(circuit))
    expect = get_unitary(convert_circuit(transpiled))
    assert np.allclose(
        target / (target[0, 0] / abs(target[0, 0])),
        expect / (expect[0, 0] / abs(expect[0, 0])),
    )


def test_convert_circuit_with_measurement() -> None:
    circuit = QuantumCircuit(3, cbit_count=2)
    original_gates = [
        gates.X(1),
        gates.H(2),
        gates.CNOT(0, 2),
        gates.RX(0, 0.125),
        gates.Measurement([0], [1]),
        gates.X(1),
    ]
    for gate in original_gates:
        circuit.add_gate(gate)

    converted = convert_circuit(circuit)
    assert converted.num_qubits == 3

    expected = QiskitQuantumCircuit(3, 2)
    expected.x(1)
    expected.h(2)
    expected.cx(0, 2)
    expected.rx(0.125, 0)
    expected.measure(0, 1)
    expected.x(1)

    # Gate order and number of registers remain same.
    assert qasm3.dumps(converted) == qasm3.dumps(expected)
