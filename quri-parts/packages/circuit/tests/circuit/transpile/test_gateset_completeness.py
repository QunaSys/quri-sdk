# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from numpy.typing import NDArray

import quri_parts.circuit.transpile.rz2hst as rz2hst
from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    QuantumCircuit,
    gate_names,
    gates,
)
from quri_parts.circuit.gate_names import CliffordGateNameType, GateNameType
from quri_parts.circuit.transpile import (
    CliffordConversionTranspiler,
    GateSetConversionTranspiler,
)


def _gate_kinds(circuit: ImmutableQuantumCircuit) -> set[str]:
    return {gate.name for gate in circuit.gates}


def _circuit_unitary(circuit: ImmutableQuantumCircuit) -> NDArray[np.complex128]:
    qulacs = pytest.importorskip("qulacs")
    qulacs_circuit = pytest.importorskip("quri_parts.qulacs.circuit")

    n = circuit.qubit_count
    qc = qulacs_circuit.convert_circuit(circuit)
    dim = 1 << n
    unitary = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        state = qulacs.QuantumState(n)
        state.set_computational_basis(i)
        qc.update_quantum_state(state)
        unitary[:, i] = state.get_vector()
    return unitary


def _assert_equivalent(
    original: ImmutableQuantumCircuit, transpiled: ImmutableQuantumCircuit
) -> None:
    original_unitary = _circuit_unitary(original)
    transpiled_unitary = _circuit_unitary(transpiled)
    idx = np.unravel_index(
        int(np.argmax(np.abs(original_unitary))), original_unitary.shape
    )
    assert abs(original_unitary[idx]) > 1e-9
    phase = transpiled_unitary[idx] / original_unitary[idx]
    assert np.allclose(original_unitary * phase, transpiled_unitary, atol=1e-6)


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

_EXACT_UNIVERSAL_TARGETS: list[list[GateNameType]] = [
    [gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.CNOT],
    [gate_names.RX, gate_names.RZ, gate_names.CNOT],
    [gate_names.RY, gate_names.RZ, gate_names.CNOT],
    [gate_names.RZ, gate_names.H, gate_names.CNOT],
    [gate_names.RZ, gate_names.SqrtX, gate_names.X, gate_names.CNOT],
    [gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.CZ],
    [gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.SWAP, gate_names.CNOT],
]


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


class TestGateSetConversionCompleteness:
    @pytest.mark.parametrize("target", _EXACT_UNIVERSAL_TARGETS)
    def test_any_circuit_to_universal_set(self, target: list[GateNameType]) -> None:
        circuit = _complex_circuit()
        transpiled = GateSetConversionTranspiler(target)(circuit)
        assert _gate_kinds(transpiled) <= set(target)
        _assert_equivalent(circuit, transpiled)

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
        circuit = _complex_circuit()
        transpiled = GateSetConversionTranspiler(_CLIFFORD_T_TARGET)(circuit)
        assert _gate_kinds(transpiled) <= set(_CLIFFORD_T_TARGET)

    def test_rotation_lowered_when_non_clifford_t_gate_in_target(self) -> None:
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

    @pytest.mark.parametrize(
        "target",
        [
            [gate_names.S, gate_names.T, gate_names.X, gate_names.CNOT],
            [gate_names.H, gate_names.T, gate_names.X, gate_names.CNOT],
        ],
    )
    def test_rotation_not_lowered_when_gridsynth_output_is_not_supported(
        self, target: list[GateNameType], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            rz2hst,
            "driver_pygridsynth",
            lambda *args, **kwargs: pytest.fail("gridsynth should not be used"),
        )
        circuit = QuantumCircuit(1)
        circuit.add_RZ_gate(0, 0.3)

        with pytest.raises(ValueError, match="cannot be converted"):
            GateSetConversionTranspiler(target)(circuit)

    def test_rotation_in_target_but_t_not_in_target(self) -> None:
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
