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

import numpy as np
from cirq.circuits.circuit import Circuit
from cirq.devices.line_qubit import LineQubit
from cirq.ops.common_gates import CNOT, CZ, H, Rx, Ry, Rz, S, T, rx, ry, rz
from cirq.ops.identity import I
from cirq.ops.matrix_gates import MatrixGate
from cirq.ops.pauli_gates import X, Y, Z
from cirq.ops.raw_types import Gate, Operation
from cirq.ops.swap_gates import SWAP
from cirq.ops.three_qubit_gates import TOFFOLI

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate, gate_names
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
from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)

#: CircuitTranspiler to convert a circit configuration suitable for Cirq.
CirqTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
)


# Implementation is follwed from Qiskit (lambda → psi)
class U1(Gate):
    """Define a cirq U1 gate."""

    def __init__(self, psi: float) -> None:
        super(U1, self)
        self.psi = psi

    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> "np.typing.NDArray[np.complex_]":
        return np.array([[1, 0], [0, np.exp(self.psi * 1.0j)]])

    def _circuit_diagram_info_(self) -> str:
        return f"U1({self.psi})"


class U2(Gate):
    """Define a cirq U2 gate."""

    def __init__(self, phi: float, psi: float) -> None:
        super(U2, self)
        self.phi = phi
        self.psi = psi

    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> "np.typing.NDArray[np.complex_]":
        return np.array(
            [
                [1 / np.sqrt(2), -np.exp(self.psi * 1.0j) / np.sqrt(2)],
                [
                    np.exp(self.phi * 1.0j) / np.sqrt(2),
                    np.exp((self.phi + self.psi) * 1.0j) / np.sqrt(2),
                ],
            ]
        )

    def _circuit_diagram_info_(self) -> str:
        return f"U2({self.phi, self.psi})"


class U3(Gate):
    """Define a cirq U3 gate."""

    def __init__(self, theta: float, phi: float, psi: float) -> None:
        super(U3, self)
        self.theta = theta
        self.phi = phi
        self.psi = psi

    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> "np.typing.NDArray[np.complex_]":
        return np.array(
            [
                [
                    np.cos(self.theta / 2),
                    -np.exp(self.psi * 1.0j) * np.sin(self.theta / 2),
                ],
                [
                    np.exp(self.phi * 1.0j) * np.sin(self.theta / 2),
                    np.exp((self.phi + self.psi) * 1.0j) * np.cos(self.theta / 2),
                ],
            ]
        )

    def _circuit_diagram_info_(self) -> str:
        return f"U3({self.theta, self.phi, self.psi})"


_single_qubit_gate_cirq: Mapping[SingleQubitGateNameType, Gate] = {
    gate_names.Identity: I,
    gate_names.X: X,
    gate_names.Y: Y,
    gate_names.Z: Z,
    gate_names.H: H,
    gate_names.S: S,
    gate_names.Sdag: S**-1,
    gate_names.SqrtX: X**0.5,
    gate_names.SqrtXdag: X**-0.5,
    gate_names.SqrtY: Y**0.5,
    gate_names.SqrtYdag: Y**-0.5,
    gate_names.T: T,
    gate_names.Tdag: T**-1,
}

_single_qubit_rotation_gate_cirq: Mapping[
    SingleQubitGateNameType,
    Union[
        Callable[[float], Gate],
        Callable[[float, float], Gate],
        Callable[[float, float, float], Gate],
        Callable[[float], Rx],
        Callable[[float], Ry],
        Callable[[float], Rz],
    ],
] = {
    gate_names.U1: U1,
    gate_names.U2: U2,
    gate_names.U3: U3,
    gate_names.RX: rx,
    gate_names.RY: ry,
    gate_names.RZ: rz,
}

_two_qubit_gate_cirq: Mapping[TwoQubitGateNameType, Gate] = {
    gate_names.CNOT: CNOT,
    gate_names.CZ: CZ,
    gate_names.SWAP: SWAP,
}

_three_qubit_gate_cirq: Mapping[ThreeQubitGateNameType, Gate] = {
    gate_names.TOFFOLI: TOFFOLI,
}


def convert_gate(
    gate: QuantumGate,
) -> Operation:
    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_cirq:
            return _single_qubit_gate_cirq[gate.name].on(
                LineQubit(*gate.target_indices)
            )
        else:
            return _single_qubit_rotation_gate_cirq[gate.name](*gate.params).on(
                LineQubit(*gate.target_indices)
            )

    elif is_two_qubit_gate_name(gate.name):
        if gate.name == "SWAP":
            return _two_qubit_gate_cirq[gate.name].on(
                LineQubit(gate.target_indices[0]),
                LineQubit(gate.target_indices[1]),
            )
        else:
            return _two_qubit_gate_cirq[gate.name].on(
                LineQubit(*gate.control_indices),
                LineQubit(*gate.target_indices),
            )

    elif is_three_qubit_gate_name(gate.name):
        return _three_qubit_gate_cirq[gate.name].on(
            *(LineQubit(q) for q in (*gate.control_indices, *gate.target_indices))
        )

    elif is_unitary_matrix_gate_name(gate.name):
        return MatrixGate(np.array(gate.unitary_matrix)).on(
            *[LineQubit(i) for i in gate.target_indices]
        )

    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported")
    else:
        raise NotImplementedError(
            f"{gate.name} conversion to Cirq gate not implemented"
        )


def convert_circuit(circuit: NonParametricQuantumCircuit) -> Circuit:
    cirq_circuit = Circuit()
    for gate in circuit.gates:
        cirq_circuit.append(convert_gate(gate))
    return cirq_circuit


__all__ = ["convert_gate", "convert_circuit", "CirqTranspiler"]
