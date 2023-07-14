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
from typing import Callable, Union

from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    QuantumCircuit,
    gate_names,
    gates,
)

from .gate import QuantumGate
from .gate_names import SingleQubitGateNameType, is_single_qubit_gate_name

_single_qubit_gate_dagger: Mapping[
    SingleQubitGateNameType, Callable[[int], QuantumGate]
] = {
    gate_names.S: gates.Sdag,
    gate_names.SqrtX: gates.SqrtXdag,
    gate_names.SqrtY: gates.SqrtYdag,
    gate_names.T: gates.Tdag,
    gate_names.Sdag: gates.S,
    gate_names.SqrtXdag: gates.SqrtX,
    gate_names.SqrtYdag: gates.SqrtY,
    gate_names.Tdag: gates.T,
}

_rotation_gate_dagger: Mapping[
    SingleQubitGateNameType,
    Union[
        Callable[[int, float], QuantumGate],
        Callable[[int, float, float], QuantumGate],
        Callable[[int, float, float, float], QuantumGate],
    ],
] = {
    gate_names.RX: gates.RX,
    gate_names.RY: gates.RY,
    gate_names.RZ: gates.RZ,
    gate_names.U1: gates.U1,
    gate_names.U2: gates.U2,
    gate_names.U3: gates.U3,
}


def inverse_gate(gate: QuantumGate) -> QuantumGate:
    target_indices = gate.target_indices
    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_dagger:
            inverse_gate = _single_qubit_gate_dagger[gate.name](*target_indices)
        elif gate.name in _rotation_gate_dagger:
            param = gate.params
            inv_param = tuple((-1 * i for i in param))
            inverse_gate = _rotation_gate_dagger[gate.name](*target_indices, *inv_param)
        else:
            inverse_gate = gate
    else:
        inverse_gate = gate
    return inverse_gate


def inverse_circuit(
    circuit: NonParametricQuantumCircuit,
) -> QuantumCircuit:
    qubit_count = circuit.qubit_count

    gates_inv = []
    for gate in circuit.gates:
        gates_inv.append(inverse_gate(gate))
    gates_inv.reverse()

    return QuantumCircuit(qubit_count, gates=gates_inv)
