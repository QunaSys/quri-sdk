# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import LinearMappedParametricQuantumCircuit
from quri_parts.core.circuit import add_parametric_commuting_paulis_exp_gate
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label


def test_add_parametric_commuting_paulis_exp_gate() -> None:
    qp_operator = Operator({PAULI_IDENTITY: 0.5, pauli_label("Z1"): -0.5})

    circuit = LinearMappedParametricQuantumCircuit(2)
    z1 = circuit.add_parameter("z1")
    z2 = circuit.add_parameter("z2")
    add_parametric_commuting_paulis_exp_gate(
        circuit, {z1: 1.5, z2: -1.2}, qp_operator, coeff=2
    )

    expected_circuit = LinearMappedParametricQuantumCircuit(2)
    z1 = expected_circuit.add_parameter("z1")
    z2 = expected_circuit.add_parameter("z2")
    expected_circuit.add_ParametricPauliRotation_gate((1,), (3,), {z1: 3, z2: -2.4})

    assert circuit.gates == expected_circuit.gates
    assert (
        circuit.bind_parameters([1, 2]).gates
        == expected_circuit.bind_parameters([1, 2]).gates
    )
    assert (
        circuit.bind_parameters([-5, 3]).gates
        == expected_circuit.bind_parameters([-5, 3]).gates
    )
