# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

from qiskit.opflow import PauliOp, PauliSumOp

from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label


def operator_from_qiskit_op(pauli_operator: Union[PauliSumOp, PauliOp]) -> Operator:
    """
    Converts an :class:`PauliSumOp` (or :class:`PauliOp`) to
    :class:`Operator.
    """
    qp_op = Operator()
    coeff_list, string_list = [], []
    if isinstance(pauli_operator, PauliOp):
        string_list.append(pauli_operator.primitive.to_label())
        coeff_list.append(pauli_operator.coeff)

    elif isinstance(pauli_operator, PauliSumOp):
        for pauli_term in pauli_operator:
            string_list.append(pauli_term.to_pauli_op().primitive.to_label())
            coeff_list.append(pauli_term.coeffs[0])
    else:
        raise NotImplementedError(
            "Only PauliSumOp and PauliOp can be input."
            f"But got {type(pauli_operator)}"
        )

    for coeff, pauli_string in zip(coeff_list, string_list):
        if all(s == "I" for s in pauli_string):
            qp_op.add_term(PAULI_IDENTITY, coeff)
        else:
            pauli_string_quri_parts = ""
            for i, pauli in enumerate(pauli_string):
                if pauli != "I":
                    pauli_string_quri_parts += f"{pauli}{i} "
            qp_op.add_term(pauli_label(pauli_string_quri_parts), coeff)
    return qp_op
