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
import qiskit.circuit.library as qgate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.extensions import UnitaryGate
from qiskit.opflow import X, Y, Z
from typing_extensions import TypeAlias

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate, gate_names
from quri_parts.circuit.gate_names import (
    MultiQubitGateNameType,
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
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)

QiskitCircuitConverter: TypeAlias = Callable[
    [NonParametricQuantumCircuit, Optional[CircuitTranspiler]], QuantumCircuit
]

#: CircuitTranspiler to convert a circuit configuration suitable for Qiskit.
QiskitTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
)


_single_qubit_gate_qiskit: Mapping[SingleQubitGateNameType, Type[Gate]] = {
    gate_names.Identity: qgate.IGate,
    gate_names.X: qgate.XGate,
    gate_names.Y: qgate.YGate,
    gate_names.Z: qgate.ZGate,
    gate_names.H: qgate.HGate,
    gate_names.S: qgate.SGate,
    gate_names.Sdag: qgate.SdgGate,
    gate_names.T: qgate.TGate,
    gate_names.Tdag: qgate.TdgGate,
    gate_names.SqrtX: qgate.SXGate,
    gate_names.SqrtXdag: qgate.SXdgGate,
}

_single_qubit_rotation_gate_qiskit: Mapping[SingleQubitGateNameType, Type[Gate]] = {
    gate_names.RX: qgate.RXGate,
    gate_names.RY: qgate.RYGate,
    gate_names.RZ: qgate.RZGate,
}

_two_qubit_gate_qiskit: Mapping[TwoQubitGateNameType, Type[Gate]] = {
    gate_names.CNOT: qgate.CXGate,
    gate_names.CZ: qgate.CZGate,
    gate_names.SWAP: qgate.SwapGate,
}

_three_qubits_gate_qiskit: Mapping[ThreeQubitGateNameType, Type[Gate]] = {
    gate_names.TOFFOLI: qgate.CCXGate,
}

_multi_qubit_gate_qiskit: Mapping[MultiQubitGateNameType, Type[Gate]] = {
    gate_names.Pauli: qgate.PauliGate,
    gate_names.PauliRotation: qgate.PauliEvolutionGate,
}


_parametric_gate_qiskit: Mapping[ParametricGateNameType, Type[Gate]] = {
    gate_names.ParametricRX: qgate.RXGate,
    gate_names.ParametricRY: qgate.RYGate,
    gate_names.ParametricRZ: qgate.RZGate,
}

_special_named_gate_matrix: Mapping[
    SingleQubitGateNameType, Sequence[Sequence[complex]]
] = {
    gate_names.SqrtY: [[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]],
    gate_names.SqrtYdag: [[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 - 0.5j]],
}


def convert_gate(gate: QuantumGate) -> Gate:
    if not is_gate_name(gate.name):
        raise ValueError(f"Unknown gate name: {gate.name}")

    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_qiskit:
            return _single_qubit_gate_qiskit[gate.name]()
        elif gate.name in _single_qubit_rotation_gate_qiskit:
            return _single_qubit_rotation_gate_qiskit[gate.name](gate.params[0])
        elif gate.name == gate_names.U1:
            return qgate.PhaseGate(gate.params[0])
        elif gate.name == gate_names.U2:
            return qgate.UGate(np.pi / 2, gate.params[0], gate.params[1])
        elif gate.name == gate_names.U3:
            return qgate.UGate(*gate.params)
        elif gate.name in _special_named_gate_matrix:
            s_matrix = _special_named_gate_matrix[gate.name]
            return UnitaryGate(np.array(s_matrix))
        else:
            assert False, "Unreachable"

    elif is_two_qubit_gate_name(gate.name) and gate.name in _two_qubit_gate_qiskit:
        return _two_qubit_gate_qiskit[gate.name]()

    elif is_three_qubit_gate_name(gate.name) and gate.name in _three_qubits_gate_qiskit:
        return _three_qubits_gate_qiskit[gate.name]()

    elif is_unitary_matrix_gate_name(gate.name):
        return UnitaryGate(gate.unitary_matrix)

    elif is_multi_qubit_gate_name(gate.name) and gate.name in _multi_qubit_gate_qiskit:
        if gate.name == gate_names.Pauli:
            q_gate = qgate.PauliGate(label=None)
            pauli_str = ""
            gate_map_str = {1: "X", 2: "Y", 3: "Z"}
            for p in reversed(gate.pauli_ids):
                pauli_str += gate_map_str[p]
            q_gate.params = [pauli_str]
            return q_gate
        elif gate.name == gate_names.PauliRotation:
            operator = 1
            gate_map_op = {1: X, 2: Y, 3: Z}
            for p in reversed(gate.pauli_ids):
                operator ^= gate_map_op[p]
            return qgate.PauliEvolutionGate(operator, time=float(gate.params[0] / 2))

    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported.")
    else:
        assert False, "Unreachable"

    raise NotImplementedError(
        f"Conversion of {gate.name} to qiskit has not been implemented."
    )


def convert_circuit(
    circuit: NonParametricQuantumCircuit,
    transpiler: Optional[CircuitTranspiler] = QiskitTranspiler(),
) -> QuantumCircuit:
    if transpiler is not None:
        circuit = transpiler(circuit)

    qiskit_circuit = QuantumCircuit(circuit.qubit_count)
    for gate in circuit.gates:
        indices = (*gate.control_indices, *gate.target_indices)
        qiskit_circuit.append(convert_gate(gate), qargs=indices)
    return qiskit_circuit


__all__ = [
    "QiskitCircuitConverter",
    "QiskitTranspiler",
    "convert_gate",
    "convert_circuit",
]
