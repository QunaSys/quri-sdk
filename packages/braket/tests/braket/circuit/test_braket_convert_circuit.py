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
from typing import Callable, Type, cast

import numpy as np
from braket.circuits import Circuit, Gate, Instruction

from quri_parts.braket.circuit import convert_circuit, convert_gate
from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuit,
    gates,
)


def instruction_equal(i1: Instruction, i2: Instruction) -> bool:
    # Compare braket IR dict expression in pydantic BaseModel.
    return cast(bool, i1.to_ir() == i2.to_ir())


def circuit_equal(c1: Circuit, c2: Circuit) -> bool:
    # Compare braket IR dict expression in pydantic BaseModel.
    return cast(bool, c1.to_ir() == c2.to_ir())


single_qubit_gate_mapping: Mapping[Callable[[int], QuantumGate], Gate] = {
    gates.Identity: Gate.I(),
    gates.X: Gate.X(),
    gates.Y: Gate.Y(),
    gates.Z: Gate.Z(),
    gates.H: Gate.H(),
    gates.S: Gate.S(),
    gates.Sdag: Gate.Si(),
    gates.SqrtX: Gate.Unitary(
        np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
    ),
    gates.SqrtXdag: Gate.Unitary(
        np.array([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]])
    ),
    gates.SqrtY: Gate.Unitary(
        np.array([[0.5 + 0.5j, -0.5 - 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]])
    ),
    gates.SqrtYdag: Gate.Unitary(
        np.array([[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 - 0.5j]])
    ),
    gates.T: Gate.T(),
    gates.Tdag: Gate.Ti(),
}


def test_convert_single_qubit_gate() -> None:
    for qp_factory, braket_gate in single_qubit_gate_mapping.items():
        qp_gate = qp_factory(7)
        converted = convert_gate(qp_gate)
        expected = Instruction(braket_gate, 7)
        assert instruction_equal(converted, expected)


two_qubit_gate_mapping: Mapping[Callable[[int, int], QuantumGate], Gate] = {
    gates.CNOT: Gate.CNot(),
    gates.CZ: Gate.CZ(),
    gates.SWAP: Gate.Swap(),
}


def test_convert_two_qubit_gate() -> None:
    for qp_factory, braket_gate in two_qubit_gate_mapping.items():
        qp_gate = qp_factory(11, 7)
        converted = convert_gate(qp_gate)
        expected = Instruction(braket_gate, [11, 7])
        assert instruction_equal(converted, expected)


rotation_gate_mapping: Mapping[Callable[[int, float], QuantumGate], Type[Gate]] = {
    gates.RX: Gate.Rx,
    gates.RY: Gate.Ry,
    gates.RZ: Gate.Rz,
}


def test_convert_rotation_gate() -> None:
    for qp_factory, braket_type in rotation_gate_mapping.items():
        qp_gate = qp_factory(7, 0.125)
        converted = convert_gate(qp_gate)
        expected = Instruction(braket_type(angle=0.125), 7)
        assert instruction_equal(converted, expected)


def test_convert_unitary_matrix_gate() -> None:
    umat = ((1, 0), (0, np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4)))
    qp_gate = gates.UnitaryMatrix((7,), umat)
    converted = convert_gate(qp_gate)
    expected = Instruction(Gate.Unitary(np.array(umat)), 7)
    assert instruction_equal(converted, expected)


def test_convert_u_gate() -> None:
    lmd1 = 0.125
    theta2, lmd2 = 0.125, -0.125
    theta3, phi3, lmd3 = 0.125, -0.125, 0.625
    for qp_gate, braket_matrix in {
        gates.U1(7, lmd1): [[1.0, 0.0], [0.0, np.exp(lmd1 * 1.0j)]],
        gates.U2(7, theta2, lmd2): [
            [1.0 / np.sqrt(2.0), -np.exp(lmd2 * 1.0j) / np.sqrt(2.0)],
            [
                np.exp(theta2 * 1.0j) / np.sqrt(2.0),
                np.exp((theta2 + lmd2) * 1.0j) / np.sqrt(2.0),
            ],
        ],
        gates.U3(7, theta3, phi3, lmd3): [
            [np.cos(theta3 / 2.0), -np.exp(lmd3 * 1.0j) * np.sin(theta3 / 2.0)],
            [
                np.exp(phi3 * 1.0j) * np.sin(theta3 / 2.0),
                np.exp((phi3 + lmd3) * 1.0j) * np.cos(theta3 / 2.0),
            ],
        ],
    }.items():
        converted = convert_gate(qp_gate)
        expected = Instruction(Gate.Unitary(np.array(braket_matrix)), 7)
        assert instruction_equal(converted, expected)


def test_convert_circuit() -> None:
    circuit = QuantumCircuit(3)
    original_gates = [
        gates.X(1),
        gates.H(2),
        gates.CNOT(0, 2),
        gates.RX(0, 0.125),
    ]
    for gate in original_gates:
        circuit.add_gate(gate)

    converted = convert_circuit(circuit)
    assert converted.qubit_count == 3

    expected = Circuit().x(1).h(2).cnot(0, 2).rx(0, 0.125)
    assert circuit_equal(converted, expected)


def test_convert_bound_parametric_circuit() -> None:
    circuit = UnboundParametricQuantumCircuit(3)
    circuit.add_X_gate(1)
    circuit.add_ParametricRX_gate(0)
    circuit.add_H_gate(2)
    circuit.add_ParametricRY_gate(1)
    circuit.add_CNOT_gate(0, 2)
    circuit.add_ParametricRZ_gate(2)
    circuit.add_RX_gate(0, 0.125)
    circuit.add_Pauli_gate(
        (0, 1, 2), (1, 2, 3)
    )  # Expected to be decomposed by default transpiler.

    params = list(2 * np.pi * np.random.random(3))
    circuit_bound = circuit.bind_parameters(params)
    converted = convert_circuit(circuit_bound)
    assert converted.qubit_count == 3

    expected = (
        Circuit()
        .x(1)
        .rx(0, params[0])
        .h(2)
        .ry(1, params[1])
        .cnot(0, 2)
        .rz(2, params[2])
        .rx(0, 0.125)
        .x(0)
        .y(1)
        .z(2)
    )
    assert circuit_equal(converted, expected)


def test_convert_bound_linear_mapped_parametric_circuit() -> None:
    circuit = LinearMappedUnboundParametricQuantumCircuit(3)
    theta, phi, rot = circuit.add_parameters("theta", "phi", "rot")
    circuit.add_X_gate(1)
    circuit.add_ParametricRX_gate(0, {theta: 0.5})
    circuit.add_H_gate(2)
    circuit.add_ParametricRY_gate(1, phi)
    circuit.add_CNOT_gate(0, 2)
    circuit.add_ParametricRZ_gate(2, {theta: -0.5})
    circuit.add_RX_gate(0, 0.125)
    circuit.add_ParametricPauliRotation_gate(
        (0, 1, 2), (1, 2, 3), {rot: 0.5}
    )  # Expected to be decomposed by default transpiler.

    params = list(2 * np.pi * np.random.random(3))
    circuit_bound = circuit.bind_parameters(params)
    converted = convert_circuit(circuit_bound)
    assert converted.qubit_count == 3

    expected = (
        Circuit()
        .x(1)
        .rx(0, params[0] * 0.5)
        .h(2)
        .ry(1, params[1])
        .cnot(0, 2)
        .rz(2, params[0] * -0.5)
        .rx(0, 0.125)
        .h(0)
        .rx(1, np.pi / 2.0)
        .cnot(1, 0)
        .cnot(2, 0)
        .rz(0, params[2] * 0.5)
        .cnot(1, 0)
        .cnot(2, 0)
        .h(0)
        .rx(1, -np.pi / 2.0)
    )
    assert circuit_equal(converted, expected)
