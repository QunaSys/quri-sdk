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

from quri_parts.chem.ansatz.particle_conserving_u1 import (
    ParticleConservingU1,
    _add_controlled_ua_gate,
    _u1_ex_gate,
)
from quri_parts.circuit import LinearMappedParametricQuantumCircuit


def test_add_controlled_ua_gate() -> None:
    qubit_count = 4
    control_index = 0
    target_index = 1
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    _add_controlled_ua_gate(circuit, control_index, target_index, phi)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_CZ_gate(control_index, target_index)
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: -1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_RY_gate(target_index, -0.5 * np.pi)
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_RY_gate(target_index, 0.5 * np.pi)
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: 1.0})
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: -1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: 1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRZ_gate(control_index, {_phi: -1.0})
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
    _add_controlled_ua_gate(circuit, control_index, target_index, {phi: -1.0})
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_CZ_gate(control_index, target_index)
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: 1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_RY_gate(target_index, -0.5 * np.pi)
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_RY_gate(target_index, 0.5 * np.pi)
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: -1.0})
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: 1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRZ_gate(target_index, {_phi: -1.0})
    expected_circuit.add_CNOT_gate(control_index, target_index)
    expected_circuit.add_ParametricRZ_gate(control_index, {_phi: 1.0})
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit


def test_u1_ex_gate() -> None:
    qubit_count = 4
    layer_index = 1
    qidx_1 = 1
    qidx_2 = 2
    circuit = _u1_ex_gate(qubit_count, layer_index, qidx_1, qidx_2)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi, theta = expected_circuit.add_parameters("phi", "theta")
    _add_controlled_ua_gate(expected_circuit, qidx_1, qidx_2, phi)
    expected_circuit.add_CZ_gate(qidx_2, qidx_1)
    expected_circuit.add_ParametricRY_gate(qidx_1, theta)
    expected_circuit.add_CNOT_gate(qidx_2, qidx_1)
    expected_circuit.add_ParametricRY_gate(qidx_1, {theta: -1.0})
    expected_circuit.add_CNOT_gate(qidx_2, qidx_1)
    _add_controlled_ua_gate(expected_circuit, qidx_1, qidx_2, {phi: -1.0})
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    layer_index = 4
    qidx_1 = 0
    qidx_2 = 5
    circuit = _u1_ex_gate(qubit_count, layer_index, qidx_1, qidx_2)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi, theta = expected_circuit.add_parameters("phi", "theta")
    _add_controlled_ua_gate(expected_circuit, qidx_1, qidx_2, phi)
    expected_circuit.add_CZ_gate(qidx_2, qidx_1)
    expected_circuit.add_ParametricRY_gate(qidx_1, theta)
    expected_circuit.add_CNOT_gate(qidx_2, qidx_1)
    expected_circuit.add_ParametricRY_gate(qidx_1, {theta: -1.0})
    expected_circuit.add_CNOT_gate(qidx_2, qidx_1)
    _add_controlled_ua_gate(expected_circuit, qidx_1, qidx_2, {phi: -1.0})

    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates

    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)

    assert bound_circuit == expected_bound_circuit


def test_particle_conserving_u1() -> None:
    qubit_count = 4
    n_layers = 1
    circuit = ParticleConservingU1(qubit_count, n_layers)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    for i in range(n_layers):
        for j in range(0, qubit_count - 1, 2):
            expected_circuit.extend(_u1_ex_gate(qubit_count, i, j, j + 1))
        for j in range(1, qubit_count - 1, 2):
            expected_circuit.extend(_u1_ex_gate(qubit_count, i, j, j + 1))
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    n_layers = 4
    circuit = ParticleConservingU1(qubit_count, n_layers)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    for i in range(n_layers):
        for j in range(0, qubit_count - 1, 2):
            expected_circuit.extend(_u1_ex_gate(qubit_count, i, j, j + 1))
        for j in range(1, qubit_count - 1, 2):
            expected_circuit.extend(_u1_ex_gate(qubit_count, i, j, j + 1))
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
