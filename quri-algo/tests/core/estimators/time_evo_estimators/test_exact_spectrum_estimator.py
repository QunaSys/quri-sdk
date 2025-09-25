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
from quri_parts.core.operator import Operator, get_sparse_matrix, pauli_label
from quri_parts.core.state import quantum_state

from quri_algo.core.estimator.time_evolution.exact_spectrum import (
    ExactTimeEvolutionExpectationValueEstimator,
)
from quri_algo.problem.operators.hamiltonian import QubitHamiltonian


def test_exact_time_evolution() -> None:
    hamiltonian = Operator({pauli_label("X0"): 1})
    hamiltonian_input = QubitHamiltonian(n_qubit=1, qubit_hamiltonian=hamiltonian)
    vals, vecs = np.linalg.eigh(get_sparse_matrix(hamiltonian).toarray())

    state = quantum_state(1, bits=0b1)

    estimator = ExactTimeEvolutionExpectationValueEstimator(
        hamiltonian_input, vals, vecs
    )

    t = np.random.random()
    assert np.isclose(estimator(state, t, None).value, np.cos(t))
