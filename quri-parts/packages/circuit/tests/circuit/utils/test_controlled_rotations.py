# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.circuit import LinearMappedParametricQuantumCircuit
from quri_parts.circuit.utils.controlled_rotations import (
    add_controlled_RX_gate,
    add_controlled_RY_gate,
)


def test_add_controlled_RX() -> None:
    qubit_count = 2
    control_index = 0
    target_index = 1
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_controlled_RX_gate(circuit, control_index, target_index, {phi: 1.0})
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_RZ_gate(target_index, 0.5 * np.pi)
    expected_circuit.add_ParametricRY_gate(target_index, {_phi: 0.5})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRY_gate(target_index, {_phi: -0.5})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_RZ_gate(target_index, -0.5 * np.pi)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    control_index = 0
    target_index = 5
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_controlled_RX_gate(circuit, control_index, target_index, {phi: -2.0})
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_RZ_gate(target_index, 0.5 * np.pi)
    expected_circuit.add_ParametricRY_gate(target_index, {_phi: -1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRY_gate(target_index, {_phi: 1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_RZ_gate(target_index, -0.5 * np.pi)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit


def test_add_controlled_RY_gate() -> None:
    qubit_count = 4
    control_index = 2
    target_index = 0
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_controlled_RY_gate(circuit, control_index, target_index, theta)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    exp_theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_ParametricRY_gate(target_index, {exp_theta: 0.5})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRY_gate(target_index, {exp_theta: -0.5})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 4
    control_index = 3
    target_index = 1
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_controlled_RY_gate(circuit, control_index, target_index, {theta: 0.5})
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    exp_theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_ParametricRY_gate(target_index, {exp_theta: 0.25})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRY_gate(target_index, {exp_theta: -0.25})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    control_index = 3
    target_index = 1
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_controlled_RY_gate(circuit, target_index, target_index, {theta: -0.5})
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    exp_theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_ParametricRY_gate(target_index, {exp_theta: -0.25})
    expected_circuit.add_CNOT_gate(target_index, target_index)
    expected_circuit.add_ParametricRY_gate(target_index, {exp_theta: 0.25})
    expected_circuit.add_CNOT_gate(target_index, target_index)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
