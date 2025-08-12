# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from openfermion.ops import QubitOperator

from quri_parts.core.operator import Operator, pauli_label
from quri_parts.openfermion.operator import operator_from_openfermion_op


def test_operator_from_openfermin_op() -> None:
    openfermion_op = QubitOperator("X0 Y1 Z2", 1.0j) + QubitOperator("X2 Y1 Z0", 0.1)
    qp_op = Operator({pauli_label("X0 Y1 Z2"): 1.0j, pauli_label("X2 Y1 Z0"): 0.1})
    assert operator_from_openfermion_op(openfermion_op) == qp_op

    openfermion_op.constant = 1.0 + 0.1j
    qp_op.constant = 1.0 + 0.1j
    assert operator_from_openfermion_op(openfermion_op) == qp_op
