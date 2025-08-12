# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from quri_parts.core.operator import Operator, PauliLabel, SinglePauli

if TYPE_CHECKING:
    from openfermion.ops import QubitOperator as OpenFermionQubitOperator


def operator_from_openfermion_op(
    openfermion_op: "OpenFermionQubitOperator",
) -> Operator:
    """Converts a OpenFermion QubitOperator to a qp Operator.

    Args:
        openfermion_op: OpenFermion QubitOperator

    Returns:
        operator: :class:`~Operator`
    """
    operator = Operator()
    for pauli_product, coef in openfermion_op.terms.items():
        index_and_pauli_list = [
            (index, SinglePauli[pauli]) for index, pauli in pauli_product
        ]
        qp_pauli_label = PauliLabel(index_and_pauli_list)
        operator.add_term(qp_pauli_label, coef)
    if openfermion_op.constant:
        operator.constant = openfermion_op.constant
    return operator
