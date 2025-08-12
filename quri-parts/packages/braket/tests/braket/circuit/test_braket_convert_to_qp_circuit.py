# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Mapping, Type

import numpy as np
import pytest
from braket.circuits import Circuit, Gate, Instruction
from scipy.stats import unitary_group

from quri_parts.braket.circuit import circuit_from_braket, gate_from_braket
from quri_parts.circuit import QuantumCircuit, QuantumGate, gates

single_qubit_gate_mapping: Mapping[Callable[[int], QuantumGate], Gate] = {
    gates.Identity: Gate.I(),
    gates.X: Gate.X(),
    gates.Y: Gate.Y(),
    gates.Z: Gate.Z(),
    gates.H: Gate.H(),
    gates.S: Gate.S(),
    gates.Sdag: Gate.Si(),
    gates.T: Gate.T(),
    gates.Tdag: Gate.Ti(),
    gates.SqrtX: Gate.V(),
    gates.SqrtXdag: Gate.Vi(),
}


def test_single_qubit_gate_conversion() -> None:
    for qp_factory, braket_gate in single_qubit_gate_mapping.items():
        expected = qp_factory(7)
        braket_instruction = Instruction(braket_gate, 7)
        converted = gate_from_braket(braket_instruction)
        assert expected == converted


two_qubit_gate_mapping: Mapping[Callable[[int, int], QuantumGate], Gate] = {
    gates.CNOT: Gate.CNot(),
    gates.CZ: Gate.CZ(),
    gates.SWAP: Gate.Swap(),
}


def test_two_qubit_gate_conversion() -> None:
    for qp_factory, braket_gate in two_qubit_gate_mapping.items():
        expected = qp_factory(11, 7)
        braket_instruction = Instruction(braket_gate, [11, 7])
        converted = gate_from_braket(braket_instruction)
        assert expected == converted


def test_three_qubit_gate_conversion() -> None:
    expected = gates.TOFFOLI(11, 7, 5)
    braket_instruction = Instruction(Gate.CCNot(), [11, 7, 5])
    converted = gate_from_braket(braket_instruction)
    assert expected == converted


rotation_gate_mapping: Mapping[Callable[[int, float], QuantumGate], Type[Gate]] = {
    gates.RX: Gate.Rx,
    gates.RY: Gate.Ry,
    gates.RZ: Gate.Rz,
}


def test_rotation_gate_conversion() -> None:
    for qp_factory, braket_type in rotation_gate_mapping.items():
        expected = qp_factory(7, 0.125)
        braket_instruction = Instruction(braket_type(angle=0.125), 7)
        converted = gate_from_braket(braket_instruction)
        assert converted == expected


def test_unitary_gate_conversion() -> None:
    matrix = unitary_group.rvs(8)
    expected = gates.UnitaryMatrix([0, 4, 7], matrix)
    braket_gate = Gate.Unitary(matrix)
    converted = gate_from_braket(Instruction(braket_gate, [7, 4, 0]))
    assert converted == expected


def test_U_gate_conversion() -> None:
    expected = gates.U1(7, 0.1)

    braket_gate = Gate.PhaseShift(0.1)
    converted = gate_from_braket(Instruction(braket_gate, [7]))
    assert expected == converted

    braket_gate = Gate.U(0.0, 0.0, 0.1)
    converted = gate_from_braket(Instruction(braket_gate, [7]))
    assert expected == converted

    expected = gates.U2(7, 0.1, 0.2)

    braket_gate = Gate.U(np.pi / 2, 0.1, 0.2)
    converted = gate_from_braket(Instruction(braket_gate, [7]))
    assert expected == converted

    expected = gates.U3(7, 0.1, 0.2, 0.3)

    braket_gate = Gate.U(0.1, 0.2, 0.3)
    converted = gate_from_braket(Instruction(braket_gate, [7]))
    assert expected == converted


def test_gates_not_supported() -> None:
    braket_gate = Gate.CSwap()
    instruction = Instruction(braket_gate, [0, 1, 2])

    with pytest.raises(AssertionError, match="CSwap is not supported."):
        gate_from_braket(instruction)


def test_circuit_conversion() -> None:
    braket_circuit = (
        Circuit()
        .x(0)
        .y(1)
        .z(2)
        .h(3)
        .s(4)
        .si(5)
        .t(6)
        .ti(7)
        .cnot(0, 1)
        .cz(2, 6)
        .swap(0, 3)
        .rx(0, 0.125)
        .ry(1, 0.876)
        .rz(2, -0.997)
        .ccnot(4, 2, 6)
        .v(0)
        .vi(1)
        .phaseshift(3, 0.1)
        .u(0, 0.0, 0.0, 0.3)
        .u(2, np.pi / 2, 0.2, 0.3)
        .u(6, 0.1, 0.2, 0.3)
    )

    qp_circuit = QuantumCircuit(8)
    qp_circuit.add_X_gate(0)
    qp_circuit.add_Y_gate(1)
    qp_circuit.add_Z_gate(2)
    qp_circuit.add_H_gate(3)
    qp_circuit.add_S_gate(4)
    qp_circuit.add_Sdag_gate(5)
    qp_circuit.add_T_gate(6)
    qp_circuit.add_Tdag_gate(7)
    qp_circuit.add_CNOT_gate(0, 1)
    qp_circuit.add_CZ_gate(2, 6)
    qp_circuit.add_SWAP_gate(0, 3)
    qp_circuit.add_RX_gate(0, 0.125)
    qp_circuit.add_RY_gate(1, 0.876)
    qp_circuit.add_RZ_gate(2, -0.997)
    qp_circuit.add_TOFFOLI_gate(4, 2, 6)
    qp_circuit.add_SqrtX_gate(0)
    qp_circuit.add_SqrtXdag_gate(1)
    qp_circuit.add_U1_gate(3, 0.1)
    qp_circuit.add_U1_gate(0, 0.3)
    qp_circuit.add_U2_gate(2, 0.2, 0.3)
    qp_circuit.add_U3_gate(6, 0.1, 0.2, 0.3)

    assert qp_circuit == circuit_from_braket(braket_circuit)
