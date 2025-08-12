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

from quri_parts.chem.ansatz.particle_conserving_u2 import (
    ParticleConservingU2,
    _u2_ex_gate,
)
from quri_parts.circuit import LinearMappedParametricQuantumCircuit


def test_u2_ex_gate() -> None:
    qubit_count = 4
    layer_index = 2
    qubit_indices = (0, 1)
    circuit = _u2_ex_gate(qubit_count, layer_index, qubit_indices)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_CNOT_gate(*qubit_indices)
    expected_circuit.add_RZ_gate(qubit_indices[0], 0.5 * np.pi)
    expected_circuit.add_ParametricRY_gate(qubit_indices[0], {phi: 1.0})
    expected_circuit.add_CNOT_gate(qubit_indices[1], qubit_indices[0])
    expected_circuit.add_ParametricRY_gate(qubit_indices[0], {phi: -1.0})
    expected_circuit.add_CNOT_gate(qubit_indices[1], qubit_indices[0])
    expected_circuit.add_RZ_gate(qubit_indices[0], -0.5 * np.pi)
    expected_circuit.add_CNOT_gate(qubit_indices[0], qubit_indices[1])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    layer_index = 8
    qubit_indices = (2, 3)
    circuit = _u2_ex_gate(qubit_count, layer_index, qubit_indices)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_CNOT_gate(*qubit_indices)
    expected_circuit.add_RZ_gate(qubit_indices[0], 0.5 * np.pi)
    expected_circuit.add_ParametricRY_gate(qubit_indices[0], {phi: 1.0})
    expected_circuit.add_CNOT_gate(qubit_indices[1], qubit_indices[0])
    expected_circuit.add_ParametricRY_gate(qubit_indices[0], {phi: -1.0})
    expected_circuit.add_CNOT_gate(qubit_indices[1], qubit_indices[0])
    expected_circuit.add_RZ_gate(qubit_indices[0], -0.5 * np.pi)
    expected_circuit.add_CNOT_gate(qubit_indices[0], qubit_indices[1])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit


def test_particle_conserving_u2() -> None:
    qubit_count = 4
    n_layers = 2
    circuit = ParticleConservingU2(qubit_count, n_layers)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    for i in range(n_layers):
        for j in range(qubit_count):
            theta = expected_circuit.add_parameter(f"theta_{i}_{j}")
            expected_circuit.add_ParametricRZ_gate(j, {theta: 1.0})
        for j in range(0, qubit_count - 1, 2):
            expected_circuit.extend(_u2_ex_gate(qubit_count, i, [j, j + 1]))
        for j in range(1, qubit_count - 1, 2):
            expected_circuit.extend(_u2_ex_gate(qubit_count, i, [j, j + 1]))
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count, 0, -1)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    n_layers = 10
    circuit = ParticleConservingU2(qubit_count, n_layers)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    for i in range(n_layers):
        for j in range(qubit_count):
            theta = expected_circuit.add_parameter(f"theta_{i}_{j}")
            expected_circuit.add_ParametricRZ_gate(j, {theta: 1.0})
        for j in range(0, qubit_count - 1, 2):
            expected_circuit.extend(_u2_ex_gate(qubit_count, i, [j, j + 1]))
        for j in range(1, qubit_count - 1, 2):
            expected_circuit.extend(_u2_ex_gate(qubit_count, i, [j, j + 1]))
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count, 0, -1)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
