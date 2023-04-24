# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from cirq.devices.line_qubit import LineQubit
from cirq.ops.identity import I
from cirq.ops.linear_combinations import PauliSum
from cirq.ops.pauli_gates import X, Y, Z
from cirq.ops.pauli_string import PauliString

from quri_parts.cirq.operator import operator_from_cirq_op
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label


def test_operator_from_cirq_op() -> None:
    q0 = LineQubit(0)
    q1 = LineQubit(1)
    q2 = LineQubit(2)
    q3 = LineQubit(3)

    p_string_id: PauliString[Any] = PauliString(1, I(q0))

    expected_pstr_id = Operator({pauli_label(PAULI_IDENTITY): 1})
    assert operator_from_cirq_op(p_string_id) == expected_pstr_id

    p_string: PauliString[Any] = PauliString(1, X(q0))

    expected_pstr = Operator({pauli_label("X0"): 1})
    assert operator_from_cirq_op(p_string) == expected_pstr

    p_sum = PauliSum.from_pauli_strings(
        [
            PauliString(-1, X(q0)),
            PauliString(2, Z(q0), Z(q1), X(q2)),
            PauliString(0.5, Y(q0), Y(q1), Z(q2), Z(q3)),
        ]
    )
    expected_psum = Operator(
        {
            pauli_label("X0"): -1,
            pauli_label("Z0 Z1 X2"): 2,
            pauli_label("Y0 Y1 Z2 Z3"): 0.5,
        }
    )

    assert operator_from_cirq_op(p_sum) == expected_psum
