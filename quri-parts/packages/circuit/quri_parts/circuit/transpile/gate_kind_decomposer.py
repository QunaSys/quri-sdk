# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence

import numpy as np

from quri_parts.circuit import QuantumGate, gate_names, gates

from .transpiler import GateKindDecomposer


class CNOT2CZHTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes CNOT gates into sequences of H and
    CZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CNOT]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            gates.H(target),
            gates.CZ(control, target),
            gates.H(target),
        ]


class CZ2CNOTHTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes CZ gates into sequences of H and
    CNOT gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CZ]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            gates.H(target),
            gates.CNOT(control, target),
            gates.H(target),
        ]


class CZ2RXRYCNOTTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes CZ gates into sequences of RX, RY,
    and CNOT gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CZ]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            gates.RY(target, np.pi / 2.0),
            gates.RX(target, np.pi),
            gates.CNOT(control, target),
            gates.RY(target, np.pi / 2.0),
            gates.RX(target, np.pi),
        ]


class H2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes H gates into sequences of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.H]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, np.pi / 2.0),
        ]


class H2RXRYTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes H gates into sequences of RX and RY
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.H]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RY(target, np.pi / 2.0),
            gates.RX(target, np.pi),
        ]


class Identity2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which converts Identity gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Identity]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, 0.0)]


class RX2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes RX gates into sequences of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gates.RZ(target, np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, theta + np.pi),
            gates.SqrtX(target),
            gates.RZ(target, np.pi / 2.0),
        ]


class RY2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes RY gates into sequences of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta = gate.params[0]
        return [
            gates.SqrtX(target),
            gates.RZ(target, theta + np.pi),
            gates.SqrtX(target),
            gates.RZ(target, np.pi),
        ]


class S2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes S gates into of gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.S]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, np.pi / 2.0)]


class Sdag2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes Sdag gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Sdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, -np.pi / 2.0)]


class SqrtX2RXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which converts SqrtX gates into RX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RX(target, np.pi / 2.0)]


class SqrtX2RZHTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes SqrtX gates into sequences of RZ and
    H gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi / 2.0),
            gates.H(target),
            gates.RZ(target, -np.pi / 2.0),
        ]


class SqrtXdag2RXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which converts SqrtXdag gates into RX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtXdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RX(target, -np.pi / 2.0)]


class SqrtXdag2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes SqrtXdag gates into sequences of RZ
    and SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtXdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi),
            gates.SqrtX(target),
            gates.RZ(target, -np.pi),
        ]


class SqrtY2RYTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which converts SqrtY gates into RY gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RY(target, np.pi / 2.0)]


class SqrtY2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes SqrtY gates into sequences of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, np.pi / 2.0),
        ]


class SqrtYdag2RYTranspiler(GateKindDecomposer):
    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtYdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RY(target, -np.pi / 2.0)]


class SqrtYdag2RZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes SqrtYdag gates into sequences of RZ
    and SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SqrtYdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, -np.pi / 2.0),
        ]


class SWAP2CNOTTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes SWAP gates into sequences of CNOT
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.SWAP]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target1, target2 = gate.target_indices
        return [
            gates.CNOT(target1, target2),
            gates.CNOT(target2, target1),
            gates.CNOT(target1, target2),
        ]


class SWAPInsertionTranspiler(GateKindDecomposer):
    """CircuitTranspiler which inserts SWAP gates any time a multi-qubit gate
    is used assuming linear connectivity."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return list(gate_names.TWO_QUBIT_GATE_NAMES) + list(
            gate_names.THREE_QUBIT_GATE_NAMES
        )

    def _decompose_two_qubit_gate(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        if gate.name == gate_names.SWAP:
            index_0, index_1 = gate.target_indices
        else:
            index_0 = gate.control_indices[0]
            index_1 = gate.target_indices[0]
        index_order = index_1 > index_0
        if index_order:
            swap_sequence = [gates.SWAP(i, i + 1) for i in range(index_0, index_1 - 1)]
            new_index_0 = index_1 - 1
            new_index_1 = index_1
        else:
            swap_sequence = [gates.SWAP(i, i + 1) for i in range(index_1, index_0 - 1)]
            new_index_0 = index_0
            new_index_1 = index_0 - 1
        if gate.name == gate_names.SWAP:
            ti = [new_index_0, new_index_1]
            ci = []
        else:
            ti = [new_index_1]
            ci = [new_index_0]

        gate_seq = []
        gate_seq.extend(swap_sequence)
        gate_seq.append(
            QuantumGate(
                gate.name,
                ti,
                ci,
                gate.classical_indices,
                gate.params,
                gate.pauli_ids,
                gate.unitary_matrix,
            )
        )
        swap_sequence.reverse()
        gate_seq.extend(swap_sequence)
        return gate_seq

    def _decompose_three_qubit_gate(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        raise NotImplementedError(
            "SWAPInsertionTranspiler only supports two-qubit gates"
        )

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        if gate.name in gate_names.TWO_QUBIT_GATE_NAMES:
            return self._decompose_two_qubit_gate(gate)
        if gate.name in gate_names.THREE_QUBIT_GATE_NAMES:
            return self._decompose_three_qubit_gate(gate)
        raise ValueError(
            "SWAPInsertionTranspiler decompose method called with non-decomposable gate"
        )


class T2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes T gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.T]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, np.pi / 4.0)]


class Tdag2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes Tdag gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Tdag]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, -np.pi / 4.0)]


class TOFFOLI2HTTdagCNOTTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes TOFFOLI gates into sequences of H,
    T, TDag, and CNOT gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.TOFFOLI]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control1, control2 = gate.control_indices
        target = gate.target_indices[0]
        return [
            gates.H(target),
            gates.CNOT(control2, target),
            gates.Tdag(target),
            gates.CNOT(control1, target),
            gates.T(target),
            gates.CNOT(control2, target),
            gates.Tdag(target),
            gates.CNOT(control1, target),
            gates.T(control2),
            gates.T(target),
            gates.H(target),
            gates.CNOT(control1, control2),
            gates.T(control1),
            gates.Tdag(control2),
            gates.CNOT(control1, control2),
        ]


class U1ToRZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes U1 gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U1]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        lam = gate.params[0]
        return [gates.RZ(target, lam)]


class U2ToRZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes U2 gates into sequences of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U2]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        phi, lam = gate.params
        return [
            gates.RZ(target, lam - np.pi / 2.0),
            gates.SqrtX(target),
            gates.RZ(target, phi + np.pi / 2.0),
        ]


class U2ToRXRZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes U2 gates into sequences of RX and RZ
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U2]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        phi, lam = gate.params
        return [
            gates.RZ(target, lam - np.pi / 2.0),
            gates.RX(target, np.pi / 2.0),
            gates.RZ(target, phi + np.pi / 2.0),
        ]


class U3ToRZSqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes U3 gates into sequences of RZ and
    SqrtX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U3]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta, phi, lam = gate.params
        return [
            gates.RZ(target, lam),
            gates.SqrtX(target),
            gates.RZ(target, theta + np.pi),
            gates.SqrtX(target),
            gates.RZ(target, phi + np.pi),
        ]


class U3ToRXRZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes U3 gates into sequences of RX and RZ
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.U3]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta, phi, lam = gate.params
        return [
            gates.RZ(target, lam),
            gates.RX(target, np.pi / 2.0),
            gates.RZ(target, theta + np.pi),
            gates.RX(target, np.pi / 2.0),
            gates.RZ(target, phi + np.pi),
        ]


class X2HZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes X gates into sequences of H and Z
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.X]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.H(target),
            gates.Z(target),
            gates.H(target),
        ]


class X2RXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which converts X gates into RX gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.X]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RX(target, np.pi)]


class X2SqrtXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes X gates into sequences of SqrtX
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.X]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.SqrtX(target), gates.SqrtX(target)]


class Y2RZXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes Y gates into sequences of RZ and T
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Y]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.RZ(target, -np.pi),
            gates.X(target),
        ]


class Y2RYTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which converts Y gates into RY gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Y]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RY(target, np.pi)]


class Z2RZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes Z gates into RZ gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Z]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gates.RZ(target, np.pi)]


class Z2HXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes Z gates into sequences of H and X
    gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Z]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            gates.H(target),
            gates.X(target),
            gates.H(target),
        ]
