# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from quri_parts.circuit import QuantumCircuit, X
from quri_parts.core.operator import Operator, get_sparse_matrix, pauli_label
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.simulator import evaluate_state_to_vector
from scipy.linalg import expm
from scipy.stats import unitary_group

from quri_algo.circuit.time_evolution.exact_unitary import (
    ExactUnitaryControlledTimeEvolutionCircuitFactory,
)
from quri_algo.core.estimator.hadamard_test import remap_state_for_hadamard_test
from quri_algo.problem import QubitHamiltonian


def test_exact_unitary_time_evo_circuit() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    hamiltonian_matrix = get_sparse_matrix(hamiltonian).toarray()

    evolution_time = 10.0

    evolution_op = expm(-1j * evolution_time * hamiltonian_matrix)
    controlled_evo = np.kron(
        evolution_op, np.array([[0, 0], [0, 1]], dtype=np.complex128)
    )
    controlled_evo += np.kron(
        np.eye(4), np.array([[1, 0], [0, 0]], dtype=np.complex128)
    )

    h_input = QubitHamiltonian(2, hamiltonian)
    evo_circuit_compiler = ExactUnitaryControlledTimeEvolutionCircuitFactory(h_input)
    evo_circuit = evo_circuit_compiler(evolution_time)
    assert evo_circuit.qubit_count == 3
    assert len(evo_circuit.gates) == 1
    assert np.allclose(evo_circuit.gates[0].unitary_matrix, controlled_evo)

    vector = unitary_group.rvs(4)[:, 0]
    exact_evolved_vector = evolution_op @ vector
    exact_evolved_vector = np.kron(exact_evolved_vector, [0, 1])

    circuit_evolved_vector = evaluate_state_to_vector(
        remap_state_for_hadamard_test(
            quantum_state(2, vector=vector)
        ).with_gates_applied(QuantumCircuit(3, gates=[X(0)]) + evo_circuit)
    )
    assert np.allclose(circuit_evolved_vector.vector, exact_evolved_vector)
