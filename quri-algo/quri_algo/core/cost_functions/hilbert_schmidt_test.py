# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.estimator import Estimate, QuantumEstimator
from quri_parts.core.state import CircuitQuantumState
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator

from quri_algo.core.cost_functions.base_classes import NonLocalCostFunction
from quri_algo.core.cost_functions.utils import (
    get_hs_operator,
    prepare_circuit_hilbert_schmidt_test,
)


class HilbertSchmidtTest(NonLocalCostFunction):
    """This class implements the Hilber-Schmidt test as described in
    https://quantum-journal.org/papers/q-2019-05-13-140/.

    It must be initialized using a QURI Parts style estimator and
    optionally a parameter `alpha` which determines the weighting of the
    global and local cost functions.
    """

    def __init__(
        self, estimator: QuantumEstimator[CircuitQuantumState], alpha: float = 1.0
    ):
        self.estimator = estimator
        self.alpha = alpha

    def __call__(
        self,
        target_circuit: NonParametricQuantumCircuit,
        trial_circuit: NonParametricQuantumCircuit,
        alpha: Optional[float] = None,
    ) -> Estimate[complex]:
        """Perform the Hilbert-Schmidt test and return the cost as an estimate.

        Arguments:
        target_circuit - The target circuit for the Hilbert-Schmidt test
        trial_circuit - The trial circuit for the Hilbert-Schmidt test
        alpha - Provides an optional override to the alpha that was set at class instantiation
        Note that the cost function is invariant to swapping of target and trial circuits

        Return value:
        Hilbert-Schmidt test estimate
        """
        if alpha is None:
            alpha = self.alpha

        lattice_size = trial_circuit.qubit_count
        combined_circuit = prepare_circuit_hilbert_schmidt_test(
            target_circuit, trial_circuit
        )
        combined_operator = get_hs_operator(alpha, lattice_size)

        return self.estimator(combined_operator, combined_circuit)


def create_default_hilbert_schmidt_test() -> HilbertSchmidtTest:
    """Create a default Hilbert-Schmidt test cost function.

    Return value:     A Hilbert-Schmidt test cost function
    """
    return HilbertSchmidtTest(create_qulacs_vector_estimator(), 1.0)
