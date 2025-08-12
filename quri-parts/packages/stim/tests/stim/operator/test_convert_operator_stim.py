# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from stim import PauliString

from quri_parts.core.operator import Operator, pauli_label
from quri_parts.stim.operator import _pauli_indices, convert_operator

P_LABELS = [
    pauli_label("X0"),
    pauli_label("Y1"),
    pauli_label("Z1 X4"),
    pauli_label("X4 Y2"),
]
P_INDICES = [[1], [0, 2], [0, 3, 0, 0, 1], [0, 0, 2, 0, 1]]
COEFS = [(i + 1) * 0.1 for i in range(len(P_LABELS))]


def test_pauli_indices() -> None:
    qubit_count = 10

    for p_label, expected in zip(P_LABELS, P_INDICES):
        assert _pauli_indices(p_label, qubit_count) == expected


def test_convert_operator() -> None:
    qubit_count = 10

    op = Operator()
    for p_label, coef in zip(P_LABELS, COEFS):
        op += Operator({p_label: coef})
    op_converted = convert_operator(op, qubit_count)

    expected_terms = [
        (PauliString(p_indices), coef) for p_indices, coef in zip(P_INDICES, COEFS)
    ]
    assert len(op_converted) == len(expected_terms)
    assert op_converted == expected_terms
