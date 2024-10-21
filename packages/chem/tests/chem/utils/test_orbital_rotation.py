# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.utils.orbital_rotation import add_orbital_rotation_gate
from quri_parts.circuit import CONST, LinearMappedParametricQuantumCircuit


def test_add_orbital_rotation_gate() -> None:
    qubit_count = 4
    qubit_indices = (0, 1, 2, 3)
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_orbital_rotation_gate(circuit, qubit_indices, phi)
    expected_circuit = LinearMappedParametricQuantumCircuit(4)
    _phi = expected_circuit.add_parameter("phi")
    # single excitation (0, 2)
    expected_circuit.add_CNOT_gate(0, 2)
    expected_circuit.add_ParametricRY_gate(0, {_phi: 0.5})
    expected_circuit.add_CNOT_gate(2, 0)
    expected_circuit.add_ParametricRY_gate(0, {_phi: -0.5})
    expected_circuit.add_CNOT_gate(2, 0)
    expected_circuit.add_CNOT_gate(0, 2)
    # single excitation (1, 3)
    expected_circuit.add_CNOT_gate(1, 3)
    expected_circuit.add_ParametricRY_gate(1, {_phi: 0.5})
    expected_circuit.add_CNOT_gate(3, 1)
    expected_circuit.add_ParametricRY_gate(1, {_phi: -0.5})
    expected_circuit.add_CNOT_gate(3, 1)
    expected_circuit.add_CNOT_gate(1, 3)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_orbital_rotation_gate(circuit, qubit_indices, {phi: -2.0})
    expected_circuit = LinearMappedParametricQuantumCircuit(4)
    _phi = expected_circuit.add_parameter("phi")
    # single excitation (0, 2)
    expected_circuit.add_CNOT_gate(0, 2)
    expected_circuit.add_ParametricRY_gate(0, {_phi: -1.0})
    expected_circuit.add_CNOT_gate(2, 0)
    expected_circuit.add_ParametricRY_gate(0, {_phi: 1.0})
    expected_circuit.add_CNOT_gate(2, 0)
    expected_circuit.add_CNOT_gate(0, 2)
    # single excitation (1, 3)
    expected_circuit.add_CNOT_gate(1, 3)
    expected_circuit.add_ParametricRY_gate(1, {_phi: -1.0})
    expected_circuit.add_CNOT_gate(3, 1)
    expected_circuit.add_ParametricRY_gate(1, {_phi: 1.0})
    expected_circuit.add_CNOT_gate(3, 1)
    expected_circuit.add_CNOT_gate(1, 3)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    add_orbital_rotation_gate(circuit, qubit_indices, {CONST: 0.5})
    expected_circuit = LinearMappedParametricQuantumCircuit(4)
    expected_circuit.add_CNOT_gate(0, 2)
    expected_circuit.add_ParametricRY_gate(0, {CONST: 0.25})
    expected_circuit.add_CNOT_gate(2, 0)
    expected_circuit.add_ParametricRY_gate(0, {CONST: -0.25})
    expected_circuit.add_CNOT_gate(2, 0)
    expected_circuit.add_CNOT_gate(0, 2)
    # single excitation (1, 3)
    expected_circuit.add_CNOT_gate(1, 3)
    expected_circuit.add_ParametricRY_gate(1, {CONST: 0.25})
    expected_circuit.add_CNOT_gate(3, 1)
    expected_circuit.add_ParametricRY_gate(1, {CONST: -0.25})
    expected_circuit.add_CNOT_gate(3, 1)
    expected_circuit.add_CNOT_gate(1, 3)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
