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
from typing import Union

from qiskit import transpile
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit

from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    UnitaryMatrix,
    gate_names,
    gates,
)
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
)
from quri_parts.qiskit.circuit.gate_names import ECR, QiskitTwoQubitGateNameType

_single_qubit_gate_qiskit_quri_parts: Mapping[str, SingleQubitGateNameType] = {
    "id": gate_names.Identity,
    "x": gate_names.X,
    "y": gate_names.Y,
    "z": gate_names.Z,
    "h": gate_names.H,
    "s": gate_names.S,
    "sdg": gate_names.Sdag,
    "t": gate_names.T,
    "tdg": gate_names.Tdag,
    "sx": gate_names.SqrtX,
    "sxdg": gate_names.SqrtXdag,
}

_single_qubit_rotation_gate_qiskit_quri_parts: Mapping[str, SingleQubitGateNameType] = {
    "rx": gate_names.RX,
    "ry": gate_names.RY,
    "rz": gate_names.RZ,
}

_two_qubit_gate_qiskit_quri_parts: Mapping[
    str, Union[TwoQubitGateNameType, QiskitTwoQubitGateNameType]
] = {
    "cx": gate_names.CNOT,
    "cz": gate_names.CZ,
    "ecr": ECR,
    "swap": gate_names.SWAP,
}

_three_qubits_gate_quri_parts: Mapping[str, ThreeQubitGateNameType] = {
    "ccx": gate_names.TOFFOLI,
}

_U_gate_qiskit_quri_parts: Mapping[str, SingleQubitGateNameType] = {
    "p": gate_names.U1,
    "u1": gate_names.U1,
    "u2": gate_names.U2,
    "u3": gate_names.U3,
    "u": gate_names.U3,
}


def circuit_from_qiskit(
    qiskit_circuit: QiskitQuantumCircuit,
    pre_conversion: bool = False,
) -> ImmutableQuantumCircuit:
    """Converts a :class:`qiskit.QuantumCircuit` to
    :class:`ImmutableQuantumCircuit`.

    Args:
        qiskit_circuit: Qiskit QuantumCircuit to be converted to QURI Parts.
        pre_conversion: Convert gates not supported by QURI Parts in advance on the
            Qiskit side.
    """
    qubit_count = qiskit_circuit.num_qubits
    circuit = QuantumCircuit(qubit_count)

    if pre_conversion:
        allowed_gates = (
            _single_qubit_gate_qiskit_quri_parts.keys()
            | _single_qubit_rotation_gate_qiskit_quri_parts.keys()
            | _two_qubit_gate_qiskit_quri_parts.keys()
            | _three_qubits_gate_quri_parts.keys()
            | _U_gate_qiskit_quri_parts.keys()
        )
        qiskit_circuit = transpile(qiskit_circuit, basis_gates=allowed_gates)

    for instruction, q, r in qiskit_circuit:
        gname = instruction.name
        if gname in _single_qubit_gate_qiskit_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=_single_qubit_gate_qiskit_quri_parts[gname],
                    target_indices=(q[0]._index,),
                )
            )
        elif gname in _single_qubit_rotation_gate_qiskit_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=_single_qubit_rotation_gate_qiskit_quri_parts[gname],
                    target_indices=(q[0]._index,),
                    params=(instruction.params[0],),
                )
            )
        elif gname in _U_gate_qiskit_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=_U_gate_qiskit_quri_parts[gname],
                    target_indices=(q[0]._index,),
                    params=(*instruction.params,),
                )
            )
        elif gname in ["cx", "cz"]:
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_qiskit_quri_parts[gname],
                    target_indices=(q[1]._index,),
                    control_indices=(q[0]._index,),
                )
            )
        elif gname in ["ecr", "swap"]:
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_qiskit_quri_parts[gname],
                    target_indices=(
                        q[0]._index,
                        q[1]._index,
                    ),
                )
            )
        elif gname in _three_qubits_gate_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=_three_qubits_gate_quri_parts[gname],
                    target_indices=(q[2]._index,),
                    control_indices=(
                        q[0]._index,
                        q[1]._index,
                    ),
                )
            )
        elif gname == "measure":
            circuit.add_gate(
                gates.Measurement([i._index for i in q], [i._index for i in r])
            )
        else:
            mat = instruction.to_matrix()
            circuit.add_gate(
                UnitaryMatrix(
                    target_indices=[i._index for i in q],
                    unitary_matrix=mat,
                )
            )
    return circuit
