# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Mapping

import numpy as np
from braket.circuits import Circuit as BraketCircuit
from braket.circuits import Instruction as BraketInstruction

from quri_parts.circuit import QuantumCircuit, QuantumGate, gates

_single_qubit_gate_braket_quri_parts: Mapping[str, Callable[[int], QuantumGate]] = {
    "I": gates.Identity,
    "X": gates.X,
    "Y": gates.Y,
    "Z": gates.Z,
    "H": gates.H,
    "S": gates.S,
    "Si": gates.Sdag,
    "T": gates.T,
    "Ti": gates.Tdag,
    "V": gates.SqrtX,
    "Vi": gates.SqrtXdag,
}

_single_qubit_rotation_gate_braket_quri_parts: Mapping[
    str, Callable[[int, float], QuantumGate]
] = {
    "Rx": gates.RX,
    "Ry": gates.RY,
    "Rz": gates.RZ,
}

_two_qubit_gate_braket_quri_parts: Mapping[str, Callable[[int, int], QuantumGate]] = {
    "CNot": gates.CNOT,
    "CZ": gates.CZ,
    "Swap": gates.SWAP,
}


def gate_from_braket(braket_gate: BraketInstruction) -> QuantumGate:
    """Convert a Braket :class:`Instruction` to a QURI Parts
    :class:`~QuantumGate`."""
    gate_name = braket_gate.operator.name
    qubits = list(map(int, braket_gate.target.item_list))

    if gate_name in _single_qubit_gate_braket_quri_parts:
        assert len(qubits) == 1, f"{gate_name} is supposed to have 1 target index."
        return _single_qubit_gate_braket_quri_parts[gate_name](qubits[0])

    if gate_name in _single_qubit_rotation_gate_braket_quri_parts:
        assert len(qubits) == 1, f"{gate_name} is supposed to have 1 target index."
        angle = getattr(braket_gate.operator, "angle")
        return _single_qubit_rotation_gate_braket_quri_parts[gate_name](
            qubits[0], angle
        )

    if gate_name in _two_qubit_gate_braket_quri_parts:
        if gate_name == "Swap":
            assert (
                len(qubits) == 2
            ), f"{gate_name} is supposed to have 2 target indices."
        else:
            assert (
                len(qubits) == 2
            ), f"{gate_name} is supposed to have 1 target index and 1 control index."
        return _two_qubit_gate_braket_quri_parts[gate_name](*qubits)

    if gate_name == "CCNot":
        assert (
            len(qubits) == 3
        ), f"{gate_name} is supposed to have 2 control indices and 1 target index."
        return gates.TOFFOLI(*qubits)

    if gate_name == "Unitary":
        return gates.UnitaryMatrix(qubits[::-1], braket_gate.operator.to_matrix())

    if gate_name == "PhaseShift":
        assert len(qubits) == 1, f"{gate_name} is supposed to have 1 target index."
        return gates.U1(qubits[0], braket_gate.operator.angle)

    if gate_name == "U":
        assert len(qubits) == 1, f"{gate_name} is supposed to have 1 target index."
        theta, phi, lam = (
            braket_gate.operator.angle_1,
            braket_gate.operator.angle_2,
            braket_gate.operator.angle_3,
        )
        if theta == 0.0 and phi == 0.0:
            return gates.U1(qubits[0], lam)
        if theta == np.pi / 2:
            return gates.U2(qubits[0], phi, lam)
        return gates.U3(qubits[0], theta, phi, lam)

    assert False, f"{gate_name} is not supported."


def circuit_from_braket(braket_circuit: BraketCircuit) -> QuantumCircuit:
    """Convert a Braket :class:`Circuit` to QURI Parts
    :class:`~QuantumCircuit`."""
    # `qubit_count` is not computed by braket_circuit.qubit_count
    # because those qubit with no gate operations will not be counted.
    qubit_count = max(braket_circuit.qubits) + 1
    circuit = QuantumCircuit(qubit_count)
    for ins in braket_circuit.instructions:
        gate = gate_from_braket(ins)
        circuit.add_gate(gate)

    return circuit
