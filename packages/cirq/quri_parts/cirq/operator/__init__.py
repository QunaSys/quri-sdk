# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union

from cirq.ops.linear_combinations import PauliSum
from cirq.ops.pauli_string import PauliString

from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label


def operator_from_cirq_op(operator: Union[PauliString[Any], PauliSum]) -> Operator:
    """
    Converts an :class:`PauliString` (or :class:`PauliSum`) to
    :class:`Operator.
    """
    qp_op = Operator()
    if isinstance(operator, PauliString):
        operator = PauliSum.from_pauli_strings([operator])

    for op in operator:
        coeff = op.coefficient
        if len(op.qubits) == 0:
            qp_op.add_term(PAULI_IDENTITY, coeff)
        else:
            pauli_string_str = ""
            for qubit, pauli in op.items():
                pauli_string_str += f"{pauli}{qubit.x} "

            qp_op.add_term(pauli_label(pauli_string_str), coeff)
    return qp_op


__all__ = ["qp_operator_from_cirq_op"]
