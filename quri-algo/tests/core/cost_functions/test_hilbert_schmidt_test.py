# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

from quri_parts.circuit import QuantumCircuit

from quri_algo.core.cost_functions import HilbertSchmidtTest
from quri_algo.core.cost_functions.utils import (
    get_hs_operator,
    prepare_circuit_hilbert_schmidt_test,
)


def test_hilbert_schmidt_test() -> None:
    estimator = Mock()

    alpha = 0.7

    target_circuit = QuantumCircuit(4)
    target_circuit.add_H_gate(0)
    target_circuit.add_CNOT_gate(0, 1)
    trial_circuit = QuantumCircuit(4)
    trial_circuit.add_X_gate(0)
    trial_circuit.add_H_gate(0)
    trial_circuit.add_CNOT_gate(0, 1)
    lattice_size = trial_circuit.qubit_count

    expected_hs_operator = get_hs_operator(alpha, lattice_size)
    expected_circuit = prepare_circuit_hilbert_schmidt_test(
        target_circuit, trial_circuit
    ).circuit
    expected_args_estimator = [expected_hs_operator, expected_circuit]

    hstest = HilbertSchmidtTest(estimator, alpha=alpha)

    hstest(target_circuit, trial_circuit)
    call_estimator = estimator.call_args
    actual_args_estimator = [call_estimator[0][0], call_estimator[0][1].circuit]

    estimator.assert_called_once()
    for arg, arg_exp in zip(actual_args_estimator, expected_args_estimator):
        assert arg == arg_exp
