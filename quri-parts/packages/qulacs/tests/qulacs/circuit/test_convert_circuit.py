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
from typing import Callable, cast

import numpy as np
import qulacs

from quri_parts.circuit import (
    LinearMappedParametricQuantumCircuit,
    ParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    gates,
)
from quri_parts.circuit.transpile import (
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    TwoQubitUnitaryMatrixKAKTranspiler,
)
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.circuit import (
    _dense_matrix_gate_qulacs,
    convert_circuit,
    convert_gate,
    convert_parametric_circuit,
)
from quri_parts.qulacs.simulator import evaluate_state_to_vector


def gates_equal(g1: qulacs.QuantumGateBase, g2: qulacs.QuantumGateBase) -> bool:
    def gate_info(
        g: qulacs.QuantumGateBase,
    ) -> tuple[str, list[int], list[int]]:
        return (
            g.get_name(),
            g.get_target_index_list(),
            g.get_control_index_list(),
        )

    return (gate_info(g1) == gate_info(g2)) and cast(
        bool, np.all(g1.get_matrix() == g2.get_matrix())
    )


single_qubit_gate_mapping: Mapping[
    Callable[[int], QuantumGate], Callable[[int], qulacs.QuantumGateBase]
] = {
    gates.Identity: qulacs.gate.Identity,
    gates.X: qulacs.gate.X,
    gates.Y: qulacs.gate.Y,
    gates.Z: qulacs.gate.Z,
    gates.H: qulacs.gate.H,
    gates.S: qulacs.gate.S,
    gates.Sdag: qulacs.gate.Sdag,
    gates.SqrtX: qulacs.gate.sqrtX,
    gates.SqrtXdag: qulacs.gate.sqrtXdag,
    gates.SqrtY: qulacs.gate.sqrtY,
    gates.SqrtYdag: qulacs.gate.sqrtYdag,
    gates.T: qulacs.gate.T,
    gates.Tdag: qulacs.gate.Tdag,
}


def test_convert_single_qubit_gate() -> None:
    for qp_fac, qs_gate in single_qubit_gate_mapping.items():
        g = qp_fac(7)
        converted = convert_gate(g)
        expected = qs_gate(7)
        assert gates_equal(converted, expected)


two_qubit_gate_mapping: Mapping[
    Callable[[int, int], QuantumGate], Callable[[int, int], qulacs.QuantumGateBase]
] = {
    gates.CNOT: qulacs.gate.CNOT,
    gates.CZ: qulacs.gate.CZ,
    gates.SWAP: qulacs.gate.SWAP,
}


def test_convert_two_qubit_gate() -> None:
    for qp_fac, qs_gate in two_qubit_gate_mapping.items():
        g = qp_fac(11, 7)
        converted = convert_gate(g)
        expected = qs_gate(11, 7)
        assert gates_equal(converted, expected)


three_qubit_gate_mapping: Mapping[
    Callable[[int, int, int], QuantumGate],
    Callable[[int, int, int], qulacs.QuantumGateBase],
] = {
    gates.TOFFOLI: qulacs.gate.TOFFOLI,
}


def test_convert_three_qubit_gate() -> None:
    for qp_fac, qs_gate in three_qubit_gate_mapping.items():
        g = qp_fac(11, 7, 5)
        converted = convert_gate(g)
        expected = qs_gate(11, 7, 5)
        assert gates_equal(converted, expected)


rotation_gate_mapping: Mapping[
    Callable[[int, float], QuantumGate], Callable[[int, float], qulacs.QuantumGateBase]
] = {
    gates.RX: qulacs.gate.RX,
    gates.RY: qulacs.gate.RY,
    gates.RZ: qulacs.gate.RZ,
}


def test_convert_rotation_gate() -> None:
    for qp_fac, qs_gate in rotation_gate_mapping.items():
        g = qp_fac(7, 0.125)
        converted = convert_gate(g)
        expected = qs_gate(7, -0.125)
        assert gates_equal(converted, expected)

    # Check rotation angle sign
    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)

    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    assert np.allclose(
        convert_gate(gates.RX(0, np.pi / 2)).get_matrix(),
        [[c, -s * 1j], [-s * 1j, c]],
    )

    assert np.allclose(
        convert_gate(gates.RY(0, np.pi / 2)).get_matrix(),
        [[c, -s], [s, c]],
    )

    assert np.allclose(
        convert_gate(gates.RZ(0, np.pi / 2)).get_matrix(),
        [[c - s * 1j, 0], [0, c + s * 1j]],
    )


def test_convert_unitary_matrix_gate() -> None:
    umat = ((1, 0), (0, np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4)))
    expected = _dense_matrix_gate_qulacs(7, umat)
    converted = convert_gate(gates.UnitaryMatrix((7,), umat))
    assert gates_equal(converted, expected)


def test_convert_u_gate() -> None:
    for g, expected in [
        (gates.U1(7, 0.125), qulacs.gate.U1(7, 0.125)),
        (gates.U2(7, 0.125, -0.125), qulacs.gate.U2(7, 0.125, -0.125)),
        (gates.U3(7, 0.125, -0.125, 0.625), qulacs.gate.U3(7, 0.125, -0.125, 0.625)),
    ]:
        converted = convert_gate(g)
        assert gates_equal(converted, expected)


def test_convert_pauli_gate() -> None:
    g = gates.Pauli((11, 7, 13), (2, 3, 1))
    converted = convert_gate(g)
    expected = qulacs.gate.Pauli([11, 7, 13], [2, 3, 1])
    assert gates_equal(converted, expected)


def test_convert_pauli_rotation_gate() -> None:
    g = gates.PauliRotation((11, 7, 13), (2, 3, 1), 0.125)
    converted = convert_gate(g)
    expected = qulacs.gate.PauliRotation([11, 7, 13], [2, 3, 1], -0.125)
    assert gates_equal(converted, expected)

    # Check rotation angle sign
    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)
    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    assert np.allclose(
        convert_gate(
            gates.PauliRotation((0,), (1,), np.pi / 2)
        ).get_matrix(),  # noqa: E501
        [[c, -s * 1j], [-s * 1j, c]],
    )


def test_convert_circuit() -> None:
    circuit = QuantumCircuit(3)
    original_gates = [
        gates.Identity(0),
        gates.X(1),
        gates.H(2),
        gates.CNOT(0, 2),
        gates.RX(0, 0.125),
        gates.TOFFOLI(2, 0, 1),
    ]
    for g in original_gates:
        circuit.add_gate(g)

    converted = convert_circuit(circuit)
    assert converted.get_qubit_count() == 3

    expected_gates = [
        qulacs.gate.Identity(0),
        qulacs.gate.X(1),
        qulacs.gate.H(2),
        qulacs.gate.CNOT(0, 2),
        qulacs.gate.RX(0, -0.125),
        qulacs.gate.TOFFOLI(2, 0, 1),
    ]
    assert converted.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(converted.get_gate(i), expected)


def test_convert_parametric_circuit() -> None:
    circuit = ParametricQuantumCircuit(3)
    circuit.add_X_gate(1)
    circuit.add_ParametricRX_gate(0)
    circuit.add_H_gate(2)
    circuit.add_ParametricRY_gate(1)
    circuit.add_CNOT_gate(0, 2)
    circuit.add_ParametricRZ_gate(2)
    circuit.add_RX_gate(0, 0.125)
    circuit.add_ParametricPauliRotation_gate((0, 1, 2), (1, 2, 3))

    converted, param_mapper = convert_parametric_circuit(circuit)
    assert converted.get_qubit_count() == 3
    assert param_mapper((0.1, 0.2, 0.3, 0.4)) == (-0.1, -0.2, -0.3, -0.4)

    expected_gates = [
        qulacs.gate.X(1),
        qulacs.gate.ParametricRX(0, 0.0),
        qulacs.gate.H(2),
        qulacs.gate.ParametricRY(1, 0.0),
        qulacs.gate.CNOT(0, 2),
        qulacs.gate.ParametricRZ(2, 0.0),
        qulacs.gate.RX(0, -0.125),
        qulacs.gate.ParametricPauliRotation([0, 1, 2], [1, 2, 3], 0.0),
    ]
    assert converted.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(converted.get_gate(i), expected)


def test_convert_linear_mapped_parametric_circuit() -> None:
    circuit = LinearMappedParametricQuantumCircuit(3)
    theta, phi = circuit.add_parameters("theta", "phi")
    circuit.add_X_gate(1)
    circuit.add_ParametricRX_gate(0, {theta: 0.5})
    circuit.add_H_gate(2)
    circuit.add_ParametricRY_gate(1, phi)
    circuit.add_CNOT_gate(0, 2)
    circuit.add_ParametricRZ_gate(2, {theta: -0.5})
    circuit.add_RX_gate(0, 0.125)
    circuit.add_ParametricPauliRotation_gate((0, 1, 2), (1, 2, 3), {phi: 0.5})

    converted, param_mapper = convert_parametric_circuit(circuit)
    assert converted.get_qubit_count() == 3
    assert param_mapper((2.0, 0.5)) == (-1.0, -0.5, 1.0, -0.25)

    expected_gates = [
        qulacs.gate.X(1),
        qulacs.gate.ParametricRX(0, 0.0),
        qulacs.gate.H(2),
        qulacs.gate.ParametricRY(1, 0.0),
        qulacs.gate.CNOT(0, 2),
        qulacs.gate.ParametricRZ(2, 0.0),
        qulacs.gate.RX(0, -0.125),
        qulacs.gate.ParametricPauliRotation([0, 1, 2], [1, 2, 3], 0.0),
    ]
    assert converted.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(converted.get_gate(i), expected)


def test_evaluate_2qubit_unitary_matrix_gate() -> None:
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

    target = evaluate_state_to_vector(GeneralCircuitQuantumState(4, circuit)).vector
    expect = evaluate_state_to_vector(GeneralCircuitQuantumState(4, transpiled)).vector
    assert np.allclose(
        target / (target[0] / abs(target[0])), expect / (expect[0] / abs(expect[0]))
    )


def test_evaluate_1qubit_unitary_matrix_gate() -> None:
    umat = [
        [0.5644123 + 0.56382264j, 0.24297345 + 0.55182125j],
        [0.27844775 + 0.53479869j, -0.08669519 - 0.7930581j],
    ]

    circuit = QuantumCircuit(4)
    circuit.add_SingleQubitUnitaryMatrix_gate(2, umat)
    transpiled = SingleQubitUnitaryMatrix2RYRZTranspiler()(circuit)

    target = evaluate_state_to_vector(GeneralCircuitQuantumState(4, circuit)).vector
    expect = evaluate_state_to_vector(GeneralCircuitQuantumState(4, transpiled)).vector
    assert np.allclose(
        target / (target[0] / abs(target[0])), expect / (expect[0] / abs(expect[0]))
    )
