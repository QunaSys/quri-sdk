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
from typing import Sequence

from numpy import array, pi
from pytket.circuit import Circuit, Op, OpType, Unitary1qBox, Unitary2qBox, Unitary3qBox

from quri_parts.circuit import ImmutableQuantumCircuit, QuantumGate, gate_names
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
    is_unitary_matrix_gate_name,
)

_single_qubit_gate_tket: Mapping[SingleQubitGateNameType, OpType] = {
    gate_names.Identity: OpType.noop,
    gate_names.X: OpType.X,
    gate_names.Y: OpType.Y,
    gate_names.Z: OpType.Z,
    gate_names.H: OpType.H,
    gate_names.S: OpType.S,
    gate_names.Sdag: OpType.Sdg,
    gate_names.SqrtX: OpType.SX,
    gate_names.SqrtXdag: OpType.SXdg,
    gate_names.T: OpType.T,
    gate_names.Tdag: OpType.Tdg,
}

_single_qubit_rotation_gate_tket: Mapping[SingleQubitGateNameType, OpType] = {
    gate_names.U1: OpType.U1,
    gate_names.U2: OpType.U2,
    gate_names.U3: OpType.U3,
    gate_names.RX: OpType.Rx,
    gate_names.RY: OpType.Ry,
    gate_names.RZ: OpType.Rz,
}

_two_qubit_gate_tket: Mapping[TwoQubitGateNameType, OpType] = {
    gate_names.CNOT: OpType.CX,
    gate_names.CZ: OpType.CZ,
    gate_names.SWAP: OpType.SWAP,
}

_three_qubit_gate_tket: Mapping[ThreeQubitGateNameType, OpType] = {
    gate_names.TOFFOLI: OpType.CCX,
}

_special_named_gate_matrix: Mapping[
    SingleQubitGateNameType, Sequence[Sequence[complex]]
] = {
    gate_names.SqrtY: [[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]],
    gate_names.SqrtYdag: [[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 - 0.5j]],
}


def convert_gate(
    gate: QuantumGate,
) -> Op:
    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_tket:
            return Op.create(_single_qubit_gate_tket[gate.name])

        elif gate.name in _special_named_gate_matrix:
            return Unitary1qBox(array(_special_named_gate_matrix[gate.name]))

        else:
            params = [float(p / pi) for p in gate.params]
            return Op.create(_single_qubit_rotation_gate_tket[gate.name], params)

    elif is_two_qubit_gate_name(gate.name):
        return Op.create(_two_qubit_gate_tket[gate.name])

    elif is_three_qubit_gate_name(gate.name):
        return Op.create(_three_qubit_gate_tket[gate.name])

    elif is_unitary_matrix_gate_name(gate.name):
        if len(gate.target_indices) == 1:
            return Unitary1qBox(array(gate.unitary_matrix))
        elif len(gate.target_indices) == 2:
            return Unitary2qBox(array(gate.unitary_matrix))
        elif len(gate.target_indices) == 3:
            return Unitary3qBox(array(gate.unitary_matrix))
        else:
            raise ValueError("tKet only supports unitary gates up to 3 qubits")

    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported")
    else:
        raise NotImplementedError(
            f"{gate.name} conversion to tket gate not implemented"
        )


def convert_circuit(circuit: ImmutableQuantumCircuit) -> Circuit:
    tket_circuit = Circuit(circuit.qubit_count)
    for gate in circuit.gates:
        converted = convert_gate(gate)
        tket_circuit.add_gate(converted, (*gate.control_indices, *gate.target_indices))
    return tket_circuit
