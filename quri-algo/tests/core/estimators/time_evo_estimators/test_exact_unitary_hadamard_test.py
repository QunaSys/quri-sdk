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
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.sampler import create_qulacs_vector_ideal_sampler

from quri_algo.core.estimator.time_evolution import (
    ExactUnitaryTimeEvolutionHadamardTest,
)
from quri_algo.problem.operators.hamiltonian import QubitHamiltonian


def test_exact_unitary_hadamard_test() -> None:
    hamiltonian = Operator({pauli_label("X0"): 1})
    hamiltonian_input = QubitHamiltonian(n_qubit=1, qubit_hamiltonian=hamiltonian)
    sampler = create_qulacs_vector_ideal_sampler()

    state = quantum_state(1, bits=0b1)

    estimator = ExactUnitaryTimeEvolutionHadamardTest(hamiltonian_input, sampler)

    t = np.random.random()
    n_shots = np.random.randint(int(1e6))
    assert np.isclose(estimator(state, t, n_shots).value, np.cos(t))
