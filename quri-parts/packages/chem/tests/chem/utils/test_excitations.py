# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.chem.utils.excitations import (
    add_double_excitation_circuit,
    add_single_excitation_circuit,
    excitations,
    to_spin_symmetric_order,
)
from quri_parts.circuit import LinearMappedParametricQuantumCircuit
from quri_parts.circuit.utils.controlled_rotations import add_controlled_RY_gate


def test_excitations() -> None:
    assert excitations(2, 2) == ([], [])
    assert excitations(3, 2) == ([(0, 2)], [])
    assert excitations(4, 2) == ([(0, 2), (1, 3)], [(0, 1, 2, 3)])
    assert excitations(4, 2, delta_sz=-1.0) == ([(0, 3)], [])
    assert excitations(6, 2) == (
        [(0, 2), (0, 4), (1, 3), (1, 5)],
        [(0, 1, 2, 3), (0, 1, 2, 5), (0, 1, 3, 4), (0, 1, 4, 5)],
    )
    assert excitations(6, 2, delta_sz=1.0) == ([(1, 2), (1, 4)], [(0, 1, 2, 4)])
    assert excitations(6, 4) == (
        [(0, 4), (1, 5), (2, 4), (3, 5)],
        [(0, 1, 4, 5), (0, 3, 4, 5), (1, 2, 4, 5), (2, 3, 4, 5)],
    )


def test_to_spin_symmetric_order() -> None:
    assert to_spin_symmetric_order((0, 1, 4, 5)) == (0, 1, 5, 4)
    assert to_spin_symmetric_order((2, 1, 3, 6)) == (2, 1, 3, 6)
    assert to_spin_symmetric_order((0, 2, 4, 6)) == (0, 2, 6, 4)
    assert to_spin_symmetric_order((1, 3, 5, 7)) == (1, 3, 7, 5)


def test_add_single_excitation_circuit() -> None:
    qubit_count = 4
    excitation = (0, 2)
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_single_excitation_circuit(circuit, excitation, theta)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_CNOT_gate(*excitation)
    add_controlled_RY_gate(expected_circuit, excitation[1], excitation[0], _theta)
    expected_circuit.add_CNOT_gate(*excitation)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    excitation = (1, 3)
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    theta = circuit.add_parameter("theta")
    add_single_excitation_circuit(circuit, excitation, theta)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _theta = expected_circuit.add_parameter("theta")
    expected_circuit.add_CNOT_gate(*excitation)
    add_controlled_RY_gate(expected_circuit, excitation[1], excitation[0], _theta)
    expected_circuit.add_CNOT_gate(*excitation)
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit


def test_add_double_excitation_circuit() -> None:
    qubit_count = 6
    excitation = (0, 1, 2, 3)
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_double_excitation_circuit(circuit, excitation, phi)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_CNOT_gate(excitation[2], excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[2])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_H_gate(excitation[0])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[1])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: 0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: -0.125})
    expected_circuit.add_CNOT_gate(excitation[0], excitation[3])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_CNOT_gate(excitation[3], excitation[1])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: 0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: -0.125})
    expected_circuit.add_CNOT_gate(excitation[2], excitation[1])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[0])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: -0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: 0.125})
    expected_circuit.add_CNOT_gate(excitation[3], excitation[1])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[3])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: -0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: 0.125})
    expected_circuit.add_CNOT_gate(excitation[0], excitation[1])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[0])
    expected_circuit.add_H_gate(excitation[0])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[2])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[3])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit

    qubit_count = 6
    excitation = (0, 1, 4, 5)
    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    phi = circuit.add_parameter("phi")
    add_double_excitation_circuit(circuit, excitation, phi)
    expected_circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    _phi = expected_circuit.add_parameter("phi")
    expected_circuit.add_CNOT_gate(excitation[2], excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[2])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_H_gate(excitation[0])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[1])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: 0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: -0.125})
    expected_circuit.add_CNOT_gate(excitation[0], excitation[3])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_CNOT_gate(excitation[3], excitation[1])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: 0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: -0.125})
    expected_circuit.add_CNOT_gate(excitation[2], excitation[1])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[0])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: -0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: 0.125})
    expected_circuit.add_CNOT_gate(excitation[3], excitation[1])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[3])
    expected_circuit.add_ParametricRY_gate(excitation[1], {_phi: -0.125})
    expected_circuit.add_ParametricRY_gate(excitation[0], {_phi: 0.125})
    expected_circuit.add_CNOT_gate(excitation[0], excitation[1])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[0])
    expected_circuit.add_H_gate(excitation[0])
    expected_circuit.add_H_gate(excitation[3])
    expected_circuit.add_CNOT_gate(excitation[0], excitation[2])
    expected_circuit.add_CNOT_gate(excitation[2], excitation[3])
    assert circuit.parameter_count == expected_circuit.parameter_count
    assert circuit._circuit.gates == expected_circuit._circuit.gates
    params = [0.1 * (i + 1) for i in range(circuit.parameter_count)]
    bound_circuit = circuit.bind_parameters(params)
    expected_bound_circuit = expected_circuit.bind_parameters(params)
    assert bound_circuit == expected_bound_circuit
