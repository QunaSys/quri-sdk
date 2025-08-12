# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Type

import numpy as np
from braket.circuits import Circuit, Gate, Instruction
from typing_extensions import TypeAlias

from quri_parts.circuit import ImmutableQuantumCircuit, QuantumGate, gate_names
from quri_parts.circuit.gate_names import (
    ParametricGateNameType,
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_multi_qubit_gate_name,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
    is_unitary_matrix_gate_name,
)
from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    IdentityInsertionTranspiler,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)

from .braket_circuit_converter import circuit_from_braket, gate_from_braket

BraketCircuitConverter: TypeAlias = Callable[
    [ImmutableQuantumCircuit, Optional[CircuitTranspiler]], Circuit
]


class BraketSetTranspiler(SequentialTranspiler):
    """CircuitTranspiler to convert a circuit configuration suitable for
    Braket."""

    def __init__(self) -> None:
        transpilers = [
            PauliDecomposeTranspiler(),
            PauliRotationDecomposeTranspiler(),
            IdentityInsertionTranspiler(),
        ]
        super().__init__(transpilers)


_single_qubit_gate_braket: Mapping[SingleQubitGateNameType, Type[Gate]] = {
    gate_names.Identity: Gate.I,
    gate_names.X: Gate.X,
    gate_names.Y: Gate.Y,
    gate_names.Z: Gate.Z,
    gate_names.H: Gate.H,
    gate_names.S: Gate.S,
    gate_names.Sdag: Gate.Si,
    gate_names.T: Gate.T,
    gate_names.Tdag: Gate.Ti,
    gate_names.SqrtX: Gate.V,
    gate_names.SqrtXdag: Gate.Vi,
}

_single_qubit_rotation_gate_braket: Mapping[SingleQubitGateNameType, Type[Gate]] = {
    gate_names.RX: Gate.Rx,
    gate_names.RY: Gate.Ry,
    gate_names.RZ: Gate.Rz,
}

_two_qubit_gate_braket: Mapping[TwoQubitGateNameType, Type[Gate]] = {
    gate_names.CNOT: Gate.CNot,
    gate_names.CZ: Gate.CZ,
    gate_names.SWAP: Gate.Swap,
}

_three_qubit_gate_braket: Mapping[ThreeQubitGateNameType, Type[Gate]] = {
    gate_names.TOFFOLI: Gate.CCNot,
}

_parametric_gate_braket: Mapping[ParametricGateNameType, Type[Gate]] = {
    gate_names.ParametricRX: Gate.Rx,
    gate_names.ParametricRY: Gate.Ry,
    gate_names.ParametricRZ: Gate.Rz,
}

# U2 gate is implemeted as a special case of U3.
_U_gate_matrix: Mapping[SingleQubitGateNameType, Callable[[Sequence[float]], Gate]] = {
    gate_names.U1: lambda angles: Gate.PhaseShift(*angles),
    gate_names.U2: lambda angles: Gate.U(np.pi / 2, *angles),
    gate_names.U3: lambda angles: Gate.U(*angles),
}

_special_named_gate_matrix: Mapping[
    SingleQubitGateNameType, Sequence[Sequence[complex]]
] = {
    gate_names.SqrtY: [[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]],
    gate_names.SqrtYdag: [[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 - 0.5j]],
}


def convert_gate(gate: QuantumGate) -> Instruction:
    if not is_gate_name(gate.name):
        raise ValueError(f"Unknown gate name: {gate.name}")

    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_braket:
            b_gate = _single_qubit_gate_braket[gate.name](*gate.params)
            return Instruction(b_gate, gate.target_indices)
        elif gate.name in _single_qubit_rotation_gate_braket:
            b_gate = _single_qubit_rotation_gate_braket[gate.name](*gate.params)
            return Instruction(b_gate, gate.target_indices)
        elif gate.name in _U_gate_matrix:
            b_gate = _U_gate_matrix[gate.name](gate.params)
            return Instruction(b_gate, gate.target_indices)
        elif gate.name in _special_named_gate_matrix:
            s_matrix = _special_named_gate_matrix[gate.name]
            b_gate = Gate.Unitary(np.array(s_matrix))
            return Instruction(b_gate, gate.target_indices)

    elif is_two_qubit_gate_name(gate.name):
        if gate.name in _two_qubit_gate_braket:
            b_gate = _two_qubit_gate_braket[gate.name](*gate.params)
            return Instruction(
                b_gate, tuple(gate.control_indices) + tuple(gate.target_indices)
            )

    elif is_three_qubit_gate_name(gate.name):
        if gate.name in _three_qubit_gate_braket:
            b_gate = _three_qubit_gate_braket[gate.name](*gate.params)
            return Instruction(
                b_gate, tuple(gate.control_indices) + tuple(gate.target_indices)
            )

    elif is_unitary_matrix_gate_name(gate.name):
        b_gate = Gate.Unitary(np.array(gate.unitary_matrix))
        return Instruction(b_gate, reversed(gate.target_indices))

    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported.")
    elif is_multi_qubit_gate_name(gate.name):
        pass
    else:
        assert False, "Unreachable"

    raise NotImplementedError(
        f"Conversion of {gate.name} to braket has not been implemented."
    )


def convert_circuit(
    circuit: ImmutableQuantumCircuit,
    transpiler: Optional[CircuitTranspiler] = BraketSetTranspiler(),
) -> Circuit:
    if transpiler is not None:
        circuit = transpiler(circuit)

    braket_circuit = Circuit()
    for gate in circuit.gates:
        braket_circuit.add_instruction(convert_gate(gate))
    return braket_circuit


__all__ = [
    "BraketCircuitConverter",
    "BraketSetTranspiler",
    "convert_gate",
    "convert_circuit",
    "gate_from_braket",
    "circuit_from_braket",
]
