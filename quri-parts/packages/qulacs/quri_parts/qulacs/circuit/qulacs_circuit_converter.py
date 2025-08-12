# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from cmath import phase
from collections.abc import Mapping

import numpy as np
from qulacs import QuantumCircuit as QulacsQuantumCircuit

from quri_parts.circuit import ImmutableQuantumCircuit as NPQC
from quri_parts.circuit import (
    PauliRotation,
    QuantumCircuit,
    QuantumGate,
    UnitaryMatrix,
    gate_names,
)
from quri_parts.circuit.gate_names import (
    MultiQubitGateNameType,
    SingleQubitGateNameType,
    TwoQubitGateNameType,
)

_single_qubit_gate_qulacs_quri_parts: Mapping[str, SingleQubitGateNameType] = {
    "I": gate_names.Identity,
    "X": gate_names.X,
    "Y": gate_names.Y,
    "Z": gate_names.Z,
    "H": gate_names.H,
    "S": gate_names.S,
    "Sdag": gate_names.Sdag,
    "T": gate_names.T,
    "Tdag": gate_names.Tdag,
    "sqrtX": gate_names.SqrtX,
    "sqrtXdag": gate_names.SqrtXdag,
    "sqrtY": gate_names.SqrtY,
    "sqrtYdag": gate_names.SqrtYdag,
}

_single_qubit_rotation_gate_qulacs_quri_parts: Mapping[str, SingleQubitGateNameType] = {
    "X-rotation": gate_names.RX,
    "Y-rotation": gate_names.RY,
    "Z-rotation": gate_names.RZ,
}

_two_qubit_gate_qulacs_quri_parts: Mapping[str, TwoQubitGateNameType] = {
    "CNOT": gate_names.CNOT,
    "CZ": gate_names.CZ,
    "SWAP": gate_names.SWAP,
}

_multi_qubits_gate_qulacs_quri_parts: Mapping[str, MultiQubitGateNameType] = {
    "Pauli": gate_names.Pauli,
    "Pauli-rotation": gate_names.PauliRotation,
}


def circuit_from_qulacs(qulacs_circuit: QulacsQuantumCircuit) -> NPQC:
    """Converts a :class:`qulacs.QuantumCircuit` to
    :class:`ImmutableQuantumCircuit`."""
    qubit_count = qulacs_circuit.get_qubit_count()
    circuit = QuantumCircuit(qubit_count)

    num_gates = qulacs_circuit.get_gate_count()
    for i in range(num_gates):
        gate = qulacs_circuit.get_gate(i)
        gname = gate.get_name()
        if gname in _single_qubit_gate_qulacs_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=_single_qubit_gate_qulacs_quri_parts[gname],
                    target_indices=(gate.get_target_index_list()[0],),
                )
            )
        elif gname in _single_qubit_rotation_gate_qulacs_quri_parts:
            matrix = gate.get_matrix()
            if gname == "X-rotation":
                angle = phase(matrix[0][0] - matrix[1][0]) * 2
            elif gname == "Y-rotation":
                angle = phase(matrix[0][0] + matrix[1][0] * 1.0j) * 2
            elif gname == "Z-rotation":
                angle = phase(matrix[1][1] / matrix[0][0])
            else:
                raise RuntimeError(f"unknown gate: {gname}")
            circuit.add_gate(
                QuantumGate(
                    name=_single_qubit_rotation_gate_qulacs_quri_parts[gname],
                    target_indices=(gate.get_target_index_list()[0],),
                    params=(angle,),
                )
            )
        elif gname in ["CNOT", "CZ"]:
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_qulacs_quri_parts[gname],
                    target_indices=(gate.get_target_index_list()[0],),
                    control_indices=(gate.get_control_index_list()[0],),
                )
            )
        elif gname == "SWAP":
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_qulacs_quri_parts[gname],
                    target_indices=tuple(gate.get_target_index_list()),
                )
            )
        elif gname in _multi_qubits_gate_qulacs_quri_parts:
            target_indices = list()
            pauli_ids = list()
            pauli_list = json.loads(gate.to_json())["pauli"]["pauli_list"]
            for pauli in pauli_list:
                target_indices.append(int(pauli["index"]))
                pauli_ids.append(int(pauli["pauli_id"]))
            if gname == "Pauli":
                circuit.add_gate(
                    QuantumGate(
                        name=_multi_qubits_gate_qulacs_quri_parts[gname],
                        target_indices=tuple(target_indices),
                        pauli_ids=tuple(pauli_ids),
                    ),
                )
            elif gname == "Pauli-rotation":
                angle = -float(json.loads(gate.to_json())["angle"])
                circuit.add_gate(
                    PauliRotation(
                        target_indices=tuple(target_indices),
                        pauli_ids=tuple(pauli_ids),
                        angle=angle,
                    ),
                )
        elif gname == "DenseMatrix":
            mat = gate.get_matrix().astype(np.complex128)
            mat = (
                np.round(mat.real, 5).astype(np.complex128)
                + np.round(mat.imag, 5).astype(np.complex128) * 1.0j
            )
            json_dic = json.loads(gate.to_json())["control_qubit_list"]
            is_close = np.allclose(mat, np.array([[0, 1], [1, 0]], dtype=np.complex128))
            # TOFFOLI gate
            if (
                len(json_dic) == 2
                and is_close
                and json_dic[0]["value"] == "1"
                and json_dic[1]["value"] == "1"
            ):
                circuit.add_gate(
                    QuantumGate(
                        name=gate_names.TOFFOLI,
                        target_indices=(gate.get_target_index_list()[0],),
                        control_indices=(gate.get_control_index_list()),
                    ),
                )
            else:
                circuit.add_gate(
                    UnitaryMatrix(
                        target_indices=gate.get_target_index_list(),
                        unitary_matrix=mat.tolist(),
                    )
                )
        else:
            raise ValueError(f"{gname} gate is not supported.")
    return circuit
