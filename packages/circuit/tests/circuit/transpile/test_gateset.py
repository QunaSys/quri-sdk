# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np

from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    NonParametricQuantumCircuit,
    ParametricQuantumCircuit,
    ParametricQuantumCircuitProtocol,
    ParametricQuantumGate,
    QuantumCircuit,
    QuantumGate,
    gate_names,
    gates,
)
from quri_parts.circuit.gate_names import CliffordGateNameType, GateNameType
from quri_parts.circuit.transpile import (
    CliffordConversionTranspiler,
    CliffordRZSetTranspiler,
    GateSetConversionTranspiler,
    ParametricRX2RZHTranspiler,
    ParametricRY2RZHTranspiler,
    RotationConversionTranspiler,
    RX2RYRZTranspiler,
    RX2RZHTranspiler,
    RY2RXRZTranspiler,
    RY2RZHTranspiler,
    RZ2RXRYTranspiler,
)


def _gates_close(
    x: Union[QuantumGate, ParametricQuantumGate],
    y: Union[QuantumGate, ParametricQuantumGate],
) -> bool:
    if isinstance(x, ParametricQuantumGate) and isinstance(y, ParametricQuantumGate):
        return x == y
    elif isinstance(x, QuantumGate) and isinstance(y, QuantumGate):
        return (
            x.name == y.name
            and x.target_indices == y.target_indices
            and x.control_indices == y.control_indices
            and np.allclose(x.params, y.params)
            and x.pauli_ids == y.pauli_ids
            and np.allclose(x.unitary_matrix, y.unitary_matrix)
        )
    else:
        return False


def _circuit_close(
    x: Union[NonParametricQuantumCircuit, ParametricQuantumCircuitProtocol],
    y: Union[NonParametricQuantumCircuit, ParametricQuantumCircuitProtocol],
) -> bool:
    return len(x.gates) == len(y.gates) and all(
        _gates_close(a, b) for a, b in zip(x.gates, y.gates)
    )


def _gate_kinds(circuit: ImmutableQuantumCircuit) -> set[str]:
    return {gate.name for gate in circuit.gates}


def _single_qubit_clifford_circuit() -> ImmutableQuantumCircuit:
    circuit = QuantumCircuit(1)
    circuit.extend(
        [
            gates.H(0),
            gates.X(0),
            gates.SqrtX(0),
            gates.SqrtXdag(0),
            gates.Y(0),
            gates.SqrtY(0),
            gates.SqrtYdag(0),
            gates.Z(0),
            gates.S(0),
            gates.Sdag(0),
        ]
    )
    return circuit


def _clifford_and_rotation_circuit(theta: float) -> ImmutableQuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.extend(
        [
            gates.Identity(0),
            gates.H(0),
            gates.CNOT(0, 1),
            gates.RX(1, theta),
            gates.X(1),
            gates.SqrtX(1),
            gates.SqrtXdag(1),
            gates.CZ(1, 0),
            gates.RY(0, theta),
            gates.Y(0),
            gates.SqrtY(0),
            gates.SqrtYdag(0),
            gates.SWAP(0, 1),
            gates.RZ(1, theta),
            gates.Z(1),
            gates.S(1),
            gates.Sdag(1),
        ]
    )
    return circuit


def _rotation_circuit(theta: float) -> ImmutableQuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.extend(
        [
            gates.T(0),
            gates.RX(0, theta),
            gates.RY(0, theta),
            gates.RZ(0, theta),
            gates.CNOT(0, 1),
            gates.RX(1, theta),
            gates.RY(1, theta),
            gates.RZ(1, theta),
            gates.Tdag(1),
        ]
    )
    return circuit


def _complex_circuit() -> ImmutableQuantumCircuit:
    theta, phi, lam = 2.0 * np.pi * np.random.rand(3)
    umat2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    umat4 = np.identity(4, dtype=np.complex128)
    umat4[3, 3] = -1.0
    circuit = QuantumCircuit(3)
    circuit.extend(
        [
            gates.Identity(0),
            gates.H(0),
            gates.X(0),
            gates.SqrtX(0),
            gates.SqrtXdag(0),
            gates.Y(0),
            gates.SqrtY(0),
            gates.SqrtYdag(0),
            gates.Z(0),
            gates.S(0),
            gates.Sdag(0),
            gates.T(0),
            gates.Tdag(0),
            gates.RX(0, theta),
            gates.RY(0, theta),
            gates.RZ(0, theta),
            gates.U1(0, theta),
            gates.U2(0, theta, phi),
            gates.U3(0, theta, phi, lam),
            gates.CNOT(0, 1),
            gates.CZ(0, 1),
            gates.SWAP(0, 1),
            gates.TOFFOLI(0, 1, 2),
            gates.Pauli((0, 1, 2), (1, 2, 3)),
            gates.PauliRotation((0, 1, 2), (1, 2, 3), theta),
            gates.SingleQubitUnitaryMatrix(0, umat2.tolist()),
            gates.TwoQubitUnitaryMatrix(0, 1, umat4.tolist()),
        ]
    )
    return circuit


class TestCliffordConversion:
    def test_hs_transpile(self) -> None:
        target_gateset: list[CliffordGateNameType] = [gate_names.H, gate_names.S]
        circuit = _single_qubit_clifford_circuit()
        transpiled = CliffordConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_hsz_transpile(self) -> None:
        target_gateset: list[CliffordGateNameType] = [
            gate_names.H,
            gate_names.S,
            gate_names.Z,
        ]
        circuit = _single_qubit_clifford_circuit()
        transpiled = CliffordConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_xsxzs_transpile(self) -> None:
        target_gateset: list[CliffordGateNameType] = [
            gate_names.X,
            gate_names.SqrtX,
            gate_names.Z,
            gate_names.S,
        ]
        circuit = _single_qubit_clifford_circuit()
        transpiled = CliffordConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_hs_clifford_rot_transpile(self) -> None:
        target_gateset: list[CliffordGateNameType] = [gate_names.H, gate_names.S]
        circuit = _clifford_and_rotation_circuit(np.pi / 7.0)
        transpiled = CliffordConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) == {
            gate_names.Identity,
            gate_names.H,
            gate_names.S,
            gate_names.CNOT,
            gate_names.CZ,
            gate_names.SWAP,
            gate_names.RX,
            gate_names.RY,
            gate_names.RZ,
        }

    def test_hs_rot_transpile(self) -> None:
        target_gateset: list[CliffordGateNameType] = [gate_names.H, gate_names.S]
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = CliffordConversionTranspiler(target_gateset)(circuit)
        assert _circuit_close(transpiled, circuit)


class TestRotationKindDecompose:
    def test_rx2ryrz_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RX_gate(0, theta)
        transpiled = RX2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RZ(0, np.pi / 2.0),
                gates.RY(0, theta),
                gates.RZ(0, -np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_rx2rzh_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RX_gate(0, theta)
        transpiled = RX2RZHTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.H(0),
                gates.RZ(0, theta),
                gates.H(0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_ry2rxrz_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RY_gate(0, theta)
        transpiled = RY2RXRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RX(0, np.pi / 2.0),
                gates.RZ(0, theta),
                gates.RX(0, -np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_ry2rzh_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RY_gate(0, theta)
        transpiled = RY2RZHTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RZ(0, -np.pi / 2.0),
                gates.H(0),
                gates.RZ(0, theta),
                gates.H(0),
                gates.RZ(0, np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)

    def test_rz2rxry_transpile(self) -> None:
        theta = 2.0 * np.pi * np.random.rand()

        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, theta)
        transpiled = RZ2RXRYTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                gates.RX(0, np.pi / 2.0),
                gates.RY(0, -theta),
                gates.RX(0, -np.pi / 2.0),
            ]
        )
        assert _circuit_close(transpiled, expect)


class TestParametricRotationKindDecompose:
    def test_parametricrx2rzh_transpile(self) -> None:
        circuit = ParametricQuantumCircuit(1)
        circuit.add_ParametricRX_gate(0)
        transpiled = ParametricRX2RZHTranspiler()(circuit)

        expect = ParametricQuantumCircuit(1)
        expect.add_H_gate(0)
        expect.add_ParametricRZ_gate(0)
        expect.add_H_gate(0)

        assert _circuit_close(transpiled, expect)

    def test_parametricry2rzh_transpile(self) -> None:
        circuit = ParametricQuantumCircuit(1)
        circuit.add_ParametricRY_gate(0)
        transpiled = ParametricRY2RZHTranspiler()(circuit)

        expect = ParametricQuantumCircuit(1)
        expect.add_RZ_gate(0, -np.pi / 2.0)
        expect.add_H_gate(0)
        expect.add_ParametricRZ_gate(0)
        expect.add_H_gate(0)
        expect.add_RZ_gate(0, np.pi / 2.0)

        assert _circuit_close(transpiled, expect)


class TestRotationConversion:
    def test_rxryrz_transpile(self) -> None:
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = RotationConversionTranspiler(
            target_rotation=[gate_names.RX, gate_names.RY, gate_names.RZ]
        )(circuit)
        assert _circuit_close(transpiled, circuit)

    def test_rxry_transpile(self) -> None:
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = RotationConversionTranspiler(
            target_rotation=[gate_names.RX, gate_names.RY]
        )(circuit)
        assert gate_names.RZ not in _gate_kinds(transpiled)

    def test_ryrz_transpile(self) -> None:
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = RotationConversionTranspiler(
            target_rotation=[gate_names.RY, gate_names.RZ]
        )(circuit)
        assert gate_names.RX not in _gate_kinds(transpiled)

    def test_rzrx_transpile(self) -> None:
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = RotationConversionTranspiler(
            target_rotation=[gate_names.RZ, gate_names.RX]
        )(circuit)
        assert gate_names.RY not in _gate_kinds(transpiled)

    def test_rz_transpile(self) -> None:
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = RotationConversionTranspiler(
            target_rotation=[gate_names.RZ],
        )(circuit)
        gate_kinds = _gate_kinds(transpiled)
        assert not {gate_names.RX, gate_names.RY} & gate_kinds

    def test_rzh_transpile(self) -> None:
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = RotationConversionTranspiler(
            target_rotation=[gate_names.RZ],
            favorable_clifford=[gate_names.H],
        )(circuit)
        gate_kinds = _gate_kinds(transpiled)
        assert not {gate_names.RX, gate_names.RY, gate_names.SqrtX} & gate_kinds
        assert gate_names.H in gate_kinds

    def test_rzsx_transpile(self) -> None:
        circuit = _rotation_circuit(np.pi / 7.0)
        transpiled = RotationConversionTranspiler(
            target_rotation=[gate_names.RZ],
            favorable_clifford=[gate_names.SqrtX],
        )(circuit)
        gate_kinds = _gate_kinds(transpiled)
        assert not {gate_names.RX, gate_names.RY, gate_names.H} & gate_kinds
        assert gate_names.SqrtX in gate_kinds


class TestGateSetConversion:
    def test_rxryrzcx_transpile(self) -> None:
        target_gateset: list[GateNameType] = [
            gate_names.RX,
            gate_names.RY,
            gate_names.RZ,
            gate_names.CNOT,
        ]
        theta = np.pi / 7.0
        circuit = _clifford_and_rotation_circuit(theta) + _rotation_circuit(theta)
        transpiled = GateSetConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_rxryrzcx_complex_transpile(self) -> None:
        target_gateset: list[GateNameType] = [
            gate_names.RX,
            gate_names.RY,
            gate_names.RZ,
            gate_names.CNOT,
        ]
        transpiled = GateSetConversionTranspiler(target_gateset)(_complex_circuit())
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_xsxrzcx_transpile(self) -> None:
        target_gateset: list[GateNameType] = [
            gate_names.X,
            gate_names.SqrtX,
            gate_names.RZ,
            gate_names.CNOT,
        ]
        theta = np.pi / 7.0
        circuit = _clifford_and_rotation_circuit(theta) + _rotation_circuit(theta)
        transpiled = GateSetConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_xsxrzcx_complex_transpile(self) -> None:
        target_gateset: list[GateNameType] = [
            gate_names.X,
            gate_names.SqrtX,
            gate_names.RZ,
            gate_names.CNOT,
        ]
        transpiled = GateSetConversionTranspiler(target_gateset)(_complex_circuit())
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_hrzcx_transpile(self) -> None:
        target_gateset: list[GateNameType] = [
            gate_names.H,
            gate_names.RZ,
            gate_names.CNOT,
        ]
        theta = np.pi / 7.0
        circuit = _clifford_and_rotation_circuit(theta) + _rotation_circuit(theta)
        transpiled = GateSetConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_hrzcx_complex_transpile(self) -> None:
        target_gateset: list[GateNameType] = [
            gate_names.H,
            gate_names.RZ,
            gate_names.CNOT,
        ]
        transpiled = GateSetConversionTranspiler(target_gateset)(_complex_circuit())
        assert _gate_kinds(transpiled) <= set(target_gateset)

    def test_paulirotation_transpile(self) -> None:
        target_gateset: list[GateNameType] = [
            gate_names.H,
            gate_names.S,
            gate_names.RZ,
            gate_names.CNOT,
        ]

        targets = (0, 2)
        ids = (2, 2)
        theta = np.pi / 7.0

        circuit = QuantumCircuit(3)
        circuit.add_PauliRotation_gate(targets, ids, theta)
        transpiled = GateSetConversionTranspiler(target_gateset)(circuit)
        assert _gate_kinds(transpiled) <= set(target_gateset)
        assert list(transpiled.gates) == [
            gates.S(0),
            gates.S(0),
            gates.S(0),
            gates.H(0),
            gates.S(0),
            gates.S(0),
            gates.S(0),
            gates.S(2),
            gates.S(2),
            gates.S(2),
            gates.H(2),
            gates.S(2),
            gates.S(2),
            gates.S(2),
            gates.CNOT(2, 0),
            gates.RZ(0, theta),
            gates.CNOT(2, 0),
            gates.S(0),
            gates.H(0),
            gates.S(0),
            gates.S(2),
            gates.H(2),
            gates.S(2),
        ]


class TestTypicalGateSetConversion:
    def test_clifford_rz_set_transpile(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_RZ_gate(0, 0.5 * np.pi)
        circuit.add_RZ_gate(1, 0.25 * np.pi)
        circuit.add_RZ_gate(2, 0.5 * np.pi)
        circuit.add_RZ_gate(2, 0.25 * np.pi)

        transpiler = CliffordRZSetTranspiler()
        transpiled_circuit = transpiler(circuit)

        expected_gates = [
            gates.H(0),
            gates.CNOT(0, 1),
            gates.S(0),
            gates.RZ(1, 0.25 * np.pi),
            gates.RZ(2, 0.75 * np.pi),
        ]

        assert list(transpiled_circuit.gates) == expected_gates
