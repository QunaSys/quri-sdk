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
from typing import Callable, Sequence, cast

from numpy import array, pi
from pytket import Circuit, OpType, Qubit

from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    gate_names,
)
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
)

_single_qubit_gate_quri_parts: Mapping[OpType, SingleQubitGateNameType] = {
    OpType.noop: gate_names.Identity,
    OpType.X: gate_names.X,
    OpType.Y: gate_names.Y,
    OpType.Z: gate_names.Z,
    OpType.H: gate_names.H,
    OpType.S: gate_names.S,
    OpType.Sdg: gate_names.Sdag,
    OpType.SX: gate_names.SqrtX,
    OpType.SXdg: gate_names.SqrtXdag,
    OpType.T: gate_names.T,
    OpType.Tdg: gate_names.Tdag,
}

_single_qubit_rotation_gate_tket: Mapping[
    OpType,
    SingleQubitGateNameType,
] = {
    OpType.U1: gate_names.U1,
    OpType.U2: gate_names.U2,
    OpType.U3: gate_names.U3,
    OpType.Rx: gate_names.RX,
    OpType.Ry: gate_names.RY,
    OpType.Rz: gate_names.RZ,
}

_two_qubit_gate_quri_parts: Mapping[OpType, TwoQubitGateNameType] = {
    OpType.CX: gate_names.CNOT,
    OpType.CZ: gate_names.CZ,
    OpType.SWAP: gate_names.SWAP,
}

_three_qubit_gate_quri_parts: Mapping[OpType, ThreeQubitGateNameType] = {
    OpType.CCX: gate_names.TOFFOLI,
}


def circuit_from_tket(tket_circuit: Circuit) -> NonParametricQuantumCircuit:
    """Converts a :class:`pytket.Circuit` to
    :class:`~NonParametricQuantumCircuit`."""
    qubit_count = tket_circuit.n_qubits
    circuit = QuantumCircuit(qubit_count)

    for operation in tket_circuit:
        gate_name = operation.op.type
        qubit_converter: Callable[[Qubit], int] = lambda qubit: int(qubit.index[0])
        qubits = list(map(qubit_converter, cast(Sequence[Qubit], operation.qubits)))
        if gate_name in _single_qubit_gate_quri_parts:
            circuit.add_gate(
                QuantumGate(
                    name=_single_qubit_gate_quri_parts[gate_name],
                    target_indices=(qubits[0],),
                )
            )

        elif gate_name in [OpType.CX, OpType.CZ]:
            control_bit, target_bit = qubits
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_quri_parts[gate_name],
                    control_indices=(control_bit,),
                    target_indices=(target_bit,),
                )
            )

        elif gate_name == OpType.SWAP:
            target_bit_0, target_bit_1 = qubits
            circuit.add_gate(
                QuantumGate(
                    name=_two_qubit_gate_quri_parts[gate_name],
                    target_indices=(
                        target_bit_0,
                        target_bit_1,
                    ),
                )
            )

        elif gate_name in _single_qubit_rotation_gate_tket:
            parameter = array(operation.op.params) * pi
            circuit.add_gate(
                QuantumGate(
                    name=_single_qubit_rotation_gate_tket[gate_name],
                    target_indices=(qubits[0],),
                    params=tuple(parameter),
                )
            )

        elif gate_name in _three_qubit_gate_quri_parts:
            c1, c2, t = qubits
            circuit.add_gate(
                QuantumGate(
                    name=_three_qubit_gate_quri_parts[gate_name],
                    target_indices=(t,),
                    control_indices=(c1, c2),
                )
            )

        elif (
            gate_name == OpType.Unitary1qBox
            or gate_name == OpType.Unitary2qBox
            or gate_name == OpType.Unitary3qBox
        ):
            matrix = operation.op.get_matrix()
            circuit.add_UnitaryMatrix_gate(target_indices=qubits, unitary_matrix=matrix)

        else:
            raise ValueError(f"{gate_name} gate is not supported.")

    return circuit
