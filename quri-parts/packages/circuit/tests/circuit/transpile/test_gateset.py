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
import pytest
import qulacs
from numpy.typing import NDArray

import quri_parts.circuit.transpile.rz2hst as rz2hst
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
    CliffordTSetTranspiler,
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
from quri_parts.qulacs.circuit import convert_circuit


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


# Full unitary matrix of a circuit, computed with qulacs.
def _circuit_unitary(circuit: ImmutableQuantumCircuit) -> NDArray[np.complex128]:
    n = circuit.qubit_count
    qc = convert_circuit(circuit)
    dim = 1 << n
    u = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        state = qulacs.QuantumState(n)
        state.set_computational_basis(i)
        qc.update_quantum_state(state)
        u[:, i] = state.get_vector()
    return u


# Assert two circuits implement the same unitary up to a global phase.
def _assert_equivalent(
    original: ImmutableQuantumCircuit, transpiled: ImmutableQuantumCircuit
) -> None:
    u1 = _circuit_unitary(original)
    u2 = _circuit_unitary(transpiled)
    idx = np.unravel_index(int(np.argmax(np.abs(u1))), u1.shape)
    assert abs(u1[idx]) > 1e-9
    phase = u2[idx] / u1[idx]
    assert np.allclose(u1 * phase, u2, atol=1e-6)


# All pairs of single-qubit Clifford gate names that generate the full
# single-qubit Clifford group (verified by group closure).
_COMPLETE_SQ_CLIFFORD_GENERATORS: list[
    tuple[CliffordGateNameType, CliffordGateNameType]
] = [
    (gate_names.H, gate_names.S),
    (gate_names.H, gate_names.Sdag),
    (gate_names.H, gate_names.SqrtX),
    (gate_names.H, gate_names.SqrtXdag),
    (gate_names.S, gate_names.SqrtX),
    (gate_names.S, gate_names.SqrtXdag),
    (gate_names.S, gate_names.SqrtY),
    (gate_names.S, gate_names.SqrtYdag),
    (gate_names.Sdag, gate_names.SqrtX),
    (gate_names.Sdag, gate_names.SqrtXdag),
    (gate_names.Sdag, gate_names.SqrtY),
    (gate_names.Sdag, gate_names.SqrtYdag),
    (gate_names.SqrtX, gate_names.SqrtY),
    (gate_names.SqrtX, gate_names.SqrtYdag),
    (gate_names.SqrtXdag, gate_names.SqrtY),
    (gate_names.SqrtXdag, gate_names.SqrtYdag),
]

# Every single-qubit Clifford gate name.
_ALL_SINGLE_QUBIT_CLIFFORD_GATES: list[GateNameType] = [
    gate_names.H,
    gate_names.X,
    gate_names.Y,
    gate_names.Z,
    gate_names.S,
    gate_names.Sdag,
    gate_names.SqrtX,
    gate_names.SqrtXdag,
    gate_names.SqrtY,
    gate_names.SqrtYdag,
]


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
        _assert_equivalent(circuit, transpiled)


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


#: Allowed gate names in the Clifford+T gate set.
_CLIFFORD_T_GATE_NAMES = {
    gate_names.H,
    gate_names.X,
    gate_names.Y,
    gate_names.Z,
    gate_names.S,
    gate_names.Sdag,
    gate_names.SqrtX,
    gate_names.SqrtXdag,
    gate_names.SqrtY,
    gate_names.SqrtYdag,
    gate_names.CNOT,
    gate_names.CZ,
    gate_names.SWAP,
    gate_names.T,
    gate_names.Tdag,
}


class TestCliffordTSetTranspiler:
    def test_toffoli_decomposed_to_clifford_t(self) -> None:
        """A Toffoli gate is decomposed into only Clifford+T gates."""
        circuit = QuantumCircuit(3)
        circuit.add_TOFFOLI_gate(0, 1, 2)

        transpiled = CliffordTSetTranspiler()(circuit)
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES

    def test_rz_pi_over_4_becomes_t(self) -> None:
        """RZ(pi/4) is converted to a T gate."""
        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, np.pi / 4.0)

        transpiled = CliffordTSetTranspiler()(circuit)
        assert list(transpiled.gates) == [gates.T(0)]

    def test_rz_pi_over_2_becomes_s(self) -> None:
        """RZ(pi/2) is converted to an S gate."""
        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, np.pi / 2.0)

        transpiled = CliffordTSetTranspiler()(circuit)
        assert list(transpiled.gates) == [gates.S(0)]

    def test_already_clifford_t_circuit_passes_through(self) -> None:
        """A circuit already using only Clifford+T gates passes through
        unchanged."""
        circuit = QuantumCircuit(2)
        circuit.add_H_gate(0)
        circuit.add_T_gate(0)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_Tdag_gate(1)
        circuit.add_S_gate(0)

        transpiled = CliffordTSetTranspiler()(circuit)
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES
        assert list(transpiled.gates) == list(circuit.gates)

    def test_complex_circuit_only_clifford_t_gates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A complex circuit with many gate types, including arbitrary-angle
        rotations, is transpiled to only Clifford+T gates."""
        monkeypatch.setattr(
            rz2hst,
            "driver_pygridsynth",
            lambda *a, **k: lambda theta, eps: ("HSTX", float("nan")),
        )
        transpiled = CliffordTSetTranspiler()(_complex_circuit())
        # Arbitrary-angle RZ gates are approximated into {H, S, T} gates, so
        # no RZ should remain in the output.
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES
        assert gate_names.RZ not in _gate_kinds(transpiled)

    def test_arbitrary_rz_approximated_to_clifford_t(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An RZ whose angle is not a Clifford+T gate is approximated into a
        sequence of Clifford+T gates."""
        monkeypatch.setattr(
            rz2hst,
            "driver_pygridsynth",
            lambda *a, **k: lambda theta, eps: ("HSTX", float("nan")),
        )
        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, 0.3)

        transpiled = CliffordTSetTranspiler(epsilon=1e-3)(circuit)
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES
        assert gate_names.RZ not in _gate_kinds(transpiled)

    def test_clifford_gates_only_in_output(self) -> None:
        """Output of a circuit with only Clifford+T-representable rotations
        uses no RZ."""
        circuit = QuantumCircuit(2)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_RZ_gate(0, np.pi / 2.0)  # S
        circuit.add_RZ_gate(1, np.pi / 4.0)  # T
        circuit.add_RZ_gate(0, np.pi)  # Z
        circuit.add_RZ_gate(1, -np.pi / 4.0)  # Tdag

        transpiled = CliffordTSetTranspiler()(circuit)
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES
        assert gate_names.RZ not in _gate_kinds(transpiled)


class TestUnsupportedGates:
    def test_multi_controlled_gate_not_in_target_raises(self) -> None:
        # MC gates are not decomposed; validation rejects them when they are not
        # part of the target gate set.
        circuit = QuantumCircuit(3)
        circuit.add_gate(gates.MCX(2, [0, 1]))
        transpiler = GateSetConversionTranspiler(
            [gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.CNOT]
        )
        with pytest.raises(ValueError, match="cannot be converted"):
            transpiler(circuit)

    def test_multi_controlled_rotation_gate_not_in_target_raises(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_gate(
            QuantumGate(
                name=gate_names.MCRZ,
                target_indices=(2,),
                control_indices=(0, 1),
                params=(0.3,),
            )
        )
        transpiler = GateSetConversionTranspiler([gate_names.RZ, gate_names.CNOT])
        with pytest.raises(ValueError, match="cannot be converted"):
            transpiler(circuit)

    def test_multi_controlled_gate_in_target_is_kept(self) -> None:
        # MC gates are not decomposed; if they are in the target gate set they
        # are passed through unchanged.
        circuit = QuantumCircuit(3)
        circuit.add_gate(gates.MCX(2, [0, 1]))
        transpiler = GateSetConversionTranspiler([gate_names.MCX, gate_names.CNOT])
        transpiled = transpiler(circuit)
        assert [g.name for g in transpiled.gates] == [gate_names.MCX]


_CLIFFORD_T_TARGET: list[GateNameType] = [
    gate_names.H,
    gate_names.X,
    gate_names.Y,
    gate_names.Z,
    gate_names.S,
    gate_names.Sdag,
    gate_names.SqrtX,
    gate_names.SqrtXdag,
    gate_names.SqrtY,
    gate_names.SqrtYdag,
    gate_names.CNOT,
    gate_names.CZ,
    gate_names.SWAP,
    gate_names.T,
    gate_names.Tdag,
]


class TestGateSetConversionCliffordT:
    def test_toffoli_to_clifford_t(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_TOFFOLI_gate(0, 1, 2)
        transpiled = GateSetConversionTranspiler(_CLIFFORD_T_TARGET)(circuit)
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES
        # Toffoli is decomposed via TOFFOLI2CNOTTTranspiler (H, T, Tdag, CNOT).
        assert _gate_kinds(transpiled) <= {
            gate_names.H,
            gate_names.T,
            gate_names.Tdag,
            gate_names.CNOT,
        }

    def test_clifford_angle_rz_needs_no_gridsynth(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, np.pi / 4.0)
        circuit.add_RZ_gate(0, np.pi / 2.0)
        transpiled = GateSetConversionTranspiler(_CLIFFORD_T_TARGET)(circuit)
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES
        assert gate_names.RZ not in _gate_kinds(transpiled)

    @pytest.mark.gridsynth
    def test_arbitrary_rz_approximated_to_clifford_t(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, 0.3)
        transpiled = GateSetConversionTranspiler(_CLIFFORD_T_TARGET)(circuit)
        assert _gate_kinds(transpiled) <= _CLIFFORD_T_GATE_NAMES
        assert gate_names.RZ not in _gate_kinds(transpiled)


def _clifford_circuit() -> ImmutableQuantumCircuit:
    circuit = QuantumCircuit(3)
    circuit.extend(
        [
            gates.H(0),
            gates.S(0),
            gates.Sdag(1),
            gates.X(1),
            gates.Y(2),
            gates.Z(0),
            gates.SqrtX(1),
            gates.SqrtXdag(2),
            gates.SqrtY(0),
            gates.SqrtYdag(1),
            gates.CNOT(0, 1),
            gates.CZ(1, 2),
            gates.SWAP(0, 2),
            gates.H(2),
            gates.S(2),
        ]
    )
    return circuit


class TestCliffordConversionCompleteness:
    # When the target gates form a complete set of single-qubit Clifford
    # generators, CliffordConversionTranspiler converts EVERY single-qubit
    # Clifford gate into the target gate set (and the conversion is correct).
    @pytest.mark.parametrize("generators", _COMPLETE_SQ_CLIFFORD_GENERATORS)
    def test_every_single_qubit_clifford_converts(
        self, generators: tuple[CliffordGateNameType, CliffordGateNameType]
    ) -> None:
        for gate_name in _ALL_SINGLE_QUBIT_CLIFFORD_GATES:
            circuit = QuantumCircuit(1)
            circuit.add_gate(getattr(gates, gate_name)(0))
            transpiled = CliffordConversionTranspiler(list(generators))(circuit)
            assert _gate_kinds(transpiled) <= set(generators)
            _assert_equivalent(circuit, transpiled)


# Universal target gate sets needing no rotation approximation (gridsynth):
# any non-MC circuit is transpilable into them exactly.
_EXACT_UNIVERSAL_TARGETS: list[list[GateNameType]] = [
    [gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.CNOT],
    [gate_names.RX, gate_names.RZ, gate_names.CNOT],
    [gate_names.RY, gate_names.RZ, gate_names.CNOT],
    [gate_names.RZ, gate_names.H, gate_names.CNOT],
    [gate_names.RZ, gate_names.SqrtX, gate_names.X, gate_names.CNOT],
    [gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.CZ],
    [gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.SWAP, gate_names.CNOT],
]


class TestGateSetConversionCompleteness:
    # Every (non-MC) input gate is convertible into a universal target gate set.
    @pytest.mark.parametrize("target", _EXACT_UNIVERSAL_TARGETS)
    def test_any_circuit_to_universal_set(self, target: list[GateNameType]) -> None:
        circuit = _complex_circuit()
        transpiled = GateSetConversionTranspiler(target)(circuit)
        assert _gate_kinds(transpiled) <= set(target)
        _assert_equivalent(circuit, transpiled)

    # Every Clifford gate is converted to the output when the output is a
    # complete set of Clifford generators (a complete single-qubit generator
    # pair together with an entangler).
    @pytest.mark.parametrize("entangler", [gate_names.CNOT, gate_names.CZ])
    @pytest.mark.parametrize("generators", _COMPLETE_SQ_CLIFFORD_GENERATORS)
    def test_clifford_circuit_to_complete_generators(
        self,
        generators: tuple[CliffordGateNameType, CliffordGateNameType],
        entangler: GateNameType,
    ) -> None:
        target = [*generators, entangler]
        circuit = _clifford_circuit()
        transpiled = GateSetConversionTranspiler(target)(circuit)
        assert _gate_kinds(transpiled) <= set(target)
        _assert_equivalent(circuit, transpiled)

    @pytest.mark.gridsynth
    def test_complex_circuit_to_clifford_t_set(self) -> None:
        # Clifford+T is universal too; arbitrary-angle rotations in the complex
        # circuit are approximated into Clifford+T via gridsynth.
        circuit = _complex_circuit()
        transpiled = GateSetConversionTranspiler(_CLIFFORD_T_TARGET)(circuit)
        assert _gate_kinds(transpiled) <= set(_CLIFFORD_T_TARGET)

    def test_rotation_lowered_when_non_clifford_t_gate_in_target(self) -> None:
        # The target contains TOFFOLI (not a Clifford+T gate), so it is not a
        # subset of Clifford+T. Rotations must still be lowered into Clifford+T
        # rather than left undecomposed, while TOFFOLI is kept.
        target: list[GateNameType] = [
            gate_names.H,
            gate_names.S,
            gate_names.T,
            gate_names.CNOT,
            gate_names.TOFFOLI,
        ]
        circuit = QuantumCircuit(3)
        circuit.add_RZ_gate(0, np.pi / 4.0)
        circuit.add_TOFFOLI_gate(0, 1, 2)
        transpiled = GateSetConversionTranspiler(target)(circuit)
        assert _gate_kinds(transpiled) <= set(target)
        assert gate_names.RZ not in _gate_kinds(transpiled)
        assert gate_names.TOFFOLI in _gate_kinds(transpiled)

    def test_rotation_in_target_but_t_not_in_target(self) -> None:
        # A rotation gate (RZ) is in the target but T is not. T-like gates must
        # be decomposed into the rotation (T is not kept), arbitrary-angle
        # rotations are kept as RZ (no Clifford+T lowering / no gridsynth), and
        # the result is exact.
        target: list[GateNameType] = [gate_names.RZ, gate_names.H, gate_names.CNOT]
        circuit = QuantumCircuit(2)
        circuit.add_T_gate(0)
        circuit.add_Tdag_gate(1)
        circuit.add_RZ_gate(0, 0.3137)
        circuit.add_H_gate(1)
        circuit.add_CNOT_gate(0, 1)
        transpiled = GateSetConversionTranspiler(target)(circuit)
        assert _gate_kinds(transpiled) <= set(target)
        assert gate_names.T not in _gate_kinds(transpiled)
        assert gate_names.Tdag not in _gate_kinds(transpiled)
        _assert_equivalent(circuit, transpiled)
