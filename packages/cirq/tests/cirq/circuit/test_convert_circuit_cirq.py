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
from typing import Callable, Optional, Union, cast

import numpy as np
from cirq.devices.line_qubit import LineQubit
from cirq.ops.common_gates import CNOT, CZ, H, Rx, Ry, Rz, S, T, rx, ry, rz
from cirq.ops.identity import I
from cirq.ops.matrix_gates import MatrixGate
from cirq.ops.pauli_gates import X, Y, Z
from cirq.ops.raw_types import Gate, Operation, Qid
from cirq.ops.swap_gates import SWAP
from cirq.ops.three_qubit_gates import TOFFOLI
from cirq.protocols.unitary_protocol import unitary

from quri_parts.circuit import QuantumCircuit, QuantumGate, gates
from quri_parts.circuit.transpile import TwoQubitUnitaryMatrixKAKTranspiler
from quri_parts.cirq.circuit.circuit_converter import (
    U1,
    U2,
    U3,
    convert_circuit,
    convert_gate,
)


def gates_equal(g1: Operation, g2: Operation) -> bool:
    def gate_info(
        g: Operation,
    ) -> tuple[Optional[Gate], tuple[Qid, ...]]:
        return (g.gate, g.qubits)

    return (gate_info(g1) == gate_info(g2)) and cast(
        bool, np.all(unitary(g1) == unitary(g2))
    )


single_qubit_gate_mapping: Mapping[Callable[[int], QuantumGate], Gate] = {
    gates.Identity: I,
    gates.X: X,
    gates.Y: Y,
    gates.Z: Z,
    gates.H: H,
    gates.S: S,
    gates.Sdag: S**-1,
    gates.SqrtX: X**0.5,
    gates.SqrtXdag: X**-0.5,
    gates.SqrtY: Y**0.5,
    gates.SqrtYdag: Y**-0.5,
    gates.T: T,
    gates.Tdag: T**-1,
}


def test_convert_single_qubit_gate() -> None:
    for qp_fac, cirq_gate in single_qubit_gate_mapping.items():
        g = qp_fac(7)
        converted = convert_gate(g)
        expected = cirq_gate(LineQubit(7))
        assert gates_equal(converted, expected)


two_qubit_gate_mapping: Mapping[Callable[[int, int], QuantumGate], Gate] = {
    gates.CNOT: CNOT,
    gates.CZ: CZ,
    gates.SWAP: SWAP,
}


def test_convert_two_qubit_gate() -> None:
    for qp_fac, cirq_gate in two_qubit_gate_mapping.items():
        g = qp_fac(11, 7)
        converted = convert_gate(g)
        expected = cirq_gate(LineQubit(11), LineQubit(7))
        assert gates_equal(converted, expected)


three_qubit_gate_mapping: Mapping[Callable[[int, int, int], QuantumGate], Gate] = {
    gates.TOFFOLI: TOFFOLI,
}


def test_convert_three_qubit_gate() -> None:
    for qp_fac, cirq_gate in three_qubit_gate_mapping.items():
        g = qp_fac(11, 7, 5)
        converted = convert_gate(g)
        expected = cirq_gate(LineQubit(11), LineQubit(7), LineQubit(5))
        assert gates_equal(converted, expected)


rotation_gate_mapping: Mapping[
    Callable[[int, float], QuantumGate],
    Union[
        Callable[[float], Rx],
        Callable[[float], Ry],
        Callable[[float], Rz],
    ],
] = {
    gates.RX: rx,
    gates.RY: ry,
    gates.RZ: rz,
}


def test_convert_rotation_gate() -> None:
    for qp_fac, cirq_gate in rotation_gate_mapping.items():
        g = qp_fac(7, 0.125)
        converted = convert_gate(g)
        expected = cirq_gate(0.125).on(LineQubit(7))
        assert gates_equal(converted, expected)

    # Check rotation angle sign
    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)
    assert np.allclose(
        unitary(convert_gate(gates.RX(0, np.pi / 2))),
        [[c, -s * 1j], [-s * 1j, c]],
    )
    assert np.allclose(
        unitary(convert_gate(gates.RY(0, np.pi / 2))),
        [[c, -s], [s, c]],
    )
    assert np.allclose(
        unitary(convert_gate(gates.RZ(0, np.pi / 2))),
        [[c - s * 1j, 0], [0, c + s * 1j]],
    )


def test_convert_unitary_matrix_gate() -> None:
    umat = ((1, 0), (0, np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4)))
    converted = convert_gate(gates.UnitaryMatrix((7,), umat))
    expected = MatrixGate(np.array(umat)).on(LineQubit(7))
    assert np.allclose(unitary(expected), unitary(converted))


def test_convert_u_gate() -> None:
    for g, expected in [
        (gates.U1(7, 0.125), U1(0.125).on(LineQubit(7))),
        (gates.U2(7, 0.125, -0.125), U2(0.125, -0.125).on(LineQubit(7))),
        (
            gates.U3(7, 0.125, -0.125, 0.625),
            U3(0.125, -0.125, 0.625).on(LineQubit(7)),
        ),
    ]:
        converted = convert_gate(g)
        assert cast(bool, np.all(unitary(expected) == unitary(converted)))

    # Check two matrix representations explicitly
    e_1 = np.exp(np.pi / 3 * 1.0j)
    e_2 = np.exp(np.pi / 5 * 1.0j)
    s = np.sin(np.pi / 8)
    c = np.cos(np.pi / 8)
    assert np.allclose(
        unitary(convert_gate(gates.U1(0, np.pi / 3))),
        [[1, 0], [0, e_1]],
    )
    assert np.allclose(
        unitary(convert_gate(gates.U2(0, np.pi / 3, np.pi / 5))),
        [
            [1 / np.sqrt(2), -e_2 / np.sqrt(2)],
            [e_1 / np.sqrt(2), e_1 * e_2 / np.sqrt(2)],
        ],
    )
    assert np.allclose(
        unitary(convert_gate(gates.U3(0, np.pi / 4, np.pi / 3, np.pi / 5))),
        [[c, -s * e_2], [s * e_1, c * e_1 * e_2]],
    )


def test_convert_circuit() -> None:
    circuit = QuantumCircuit(3)
    original_gates = [
        gates.X(1),
        gates.H(2),
        gates.CNOT(0, 2),
        gates.RX(0, 0.125),
    ]
    for g in original_gates:
        circuit.add_gate(g)

    converted = convert_circuit(circuit)
    assert len(converted.all_qubits()) == 3

    expected_gates = [
        X(LineQubit(1)),
        H(LineQubit(2)),
        CNOT(LineQubit(0), LineQubit(2)),
        rx(0.125).on(LineQubit(0)),
    ]
    assert len([i for i in converted.all_operations()]) == len(expected_gates)
    for i, expected in enumerate(converted.all_operations()):
        assert gates_equal(expected_gates[i], expected)


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

    target = convert_circuit(circuit).unitary()
    expect = convert_circuit(transpiled).unitary()
    assert np.allclose(
        target / (target[0, 0] / abs(target[0, 0])),
        expect / (expect[0, 0] / abs(expect[0, 0])),
    )
