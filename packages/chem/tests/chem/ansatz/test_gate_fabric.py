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

from quri_parts.chem.ansatz.gate_fabric import GateFabric, _q_gate
from quri_parts.chem.utils.excitations import add_double_excitation_circuit
from quri_parts.chem.utils.orbital_rotation import add_orbital_rotation_gate
from quri_parts.circuit import CONST, LinearMappedParametricQuantumCircuit


def test_q_gate() -> None:
    qubit_count = 4
    layer_index = 2
    qubit_indices = (0, 1, 2, 3)
    circuit = _q_gate(qubit_count, layer_index, qubit_indices, False)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    theta, phi = expected_circuit.add_parameters("theta", "phi")
    add_double_excitation_circuit(expected_circuit, qubit_indices, theta)
    add_orbital_rotation_gate(expected_circuit, qubit_indices, phi)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    circuit = _q_gate(qubit_count, layer_index, qubit_indices, True)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    theta, phi = expected_circuit.add_parameters("theta", "phi")
    add_orbital_rotation_gate(expected_circuit, qubit_indices, {CONST: np.pi})
    add_double_excitation_circuit(expected_circuit, qubit_indices, theta)
    add_orbital_rotation_gate(expected_circuit, qubit_indices, phi)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit


def test_gate_fabric() -> None:
    qubit_count = 4
    n_layers = 2
    circuit = GateFabric(qubit_count, n_layers)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    for i in range(n_layers):
        for j in range(0, qubit_count - 3, 4):
            expected_circuit.extend(
                _q_gate(qubit_count, i, (j, j + 1, j + 2, j + 3), False)
            )
        for j in range(2, qubit_count - 3, 4):
            expected_circuit.extend(
                _q_gate(qubit_count, i, (j, j + 1, j + 2, j + 3), False)
            )
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    include_pi = True
    circuit = GateFabric(qubit_count, n_layers, include_pi)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    for i in range(n_layers):
        for j in range(0, qubit_count - 3, 4):
            expected_circuit.extend(
                _q_gate(qubit_count, i, (j, j + 1, j + 2, j + 3), include_pi)
            )
        for j in range(2, qubit_count - 3, 4):
            expected_circuit.extend(
                _q_gate(qubit_count, i, (j, j + 1, j + 2, j + 3), include_pi)
            )
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
