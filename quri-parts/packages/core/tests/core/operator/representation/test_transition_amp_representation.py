# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label, zero
from quri_parts.core.operator.representation import (
    transition_amp_comp_basis,
    transition_amp_representation,
)

PAULI_LABELS = [
    pauli_label({(0, 1)}),
    pauli_label({(1, 2)}),
    pauli_label({(2, 3)}),
    PAULI_IDENTITY,
]
COEFS = [0.1, 0.2j, -0.2j, 1.0]


@pytest.fixture
def operator() -> Operator:
    op = Operator()
    for p_label, coef in zip(PAULI_LABELS, COEFS):
        op[p_label] = coef
    return op


def test_transition_amp_representation() -> None:
    terms = [
        zero(),  # {}
        Operator({PAULI_IDENTITY: 10.0}),  # {0: (10.0, 0)}
        Operator({pauli_label("X1"): 1.0}),  # {2: (1.0, 0)}
        Operator({pauli_label("Y1"): 1.0}),  # {2: (-1.0j, 2)}
        Operator({pauli_label("Y3"): 3.0}),  # {8: (-3.0j, 8)}
        Operator({pauli_label("Z5"): 5.0}),  # {0: (5.0, 32)}
        Operator({pauli_label("X0 Y2 Z4"): 0.1j}),  # {5: {0.1, 20}}
        Operator({pauli_label("Y0 X2"): 0.1}),  # {5: (-0.1j, 1)}
    ]
    operator = Operator()
    for term in terms:
        operator += term
    op_bin_repr = transition_amp_representation(operator)

    expected = {
        2: [(1.0, 0), (-1.0j, 2)],
        8: [(-3.0j, 8)],
        0: [(10.0, 0), (5.0, 32)],
        5: [(0.1, 20), (-0.1j, 1)],
    }
    assert op_bin_repr == expected


def test_transition_amp_comp_basis(operator: Operator) -> None:
    op_terms = [
        Operator({PAULI_IDENTITY: 10.0}),  # {0: (10.0, 0)}
        Operator({pauli_label("X1"): 1.0}),  # {2: (1.0, 0)}
        Operator({pauli_label("Y3"): 3.0}),  # {8: (-3.0j, 8)}
        Operator({pauli_label("Z4"): 5.0}),  # {0: (5.0, 16)}
        Operator({pauli_label("X0 Y2 Z4"): 0.1j}),  # {5: {0.1, 20}}
        Operator({pauli_label("Y0 X2"): 0.1}),  # {5: (-0.1j, 1)}
    ]
    operator = Operator()
    for term in op_terms:
        operator += term

    op_bin_repr = transition_amp_representation(operator)
    assert 15 == transition_amp_comp_basis(op_bin_repr, 0, 0)
    assert 5 == transition_amp_comp_basis(op_bin_repr, 16, 16)
    assert 1 == transition_amp_comp_basis(op_bin_repr, 2, 0)
    assert 0.1 - 0.1j == transition_amp_comp_basis(op_bin_repr, 28, 25)
    assert -0.1 - 0.1j == transition_amp_comp_basis(op_bin_repr, 4, 1)
    assert -0.1 - 0.1j == transition_amp_comp_basis(op_bin_repr, 16, 21)
    assert 0.1 + 0.1j == transition_amp_comp_basis(op_bin_repr, 3, 6)
    assert -3.0j == transition_amp_comp_basis(op_bin_repr, 2, 10)
    assert 3.0j == transition_amp_comp_basis(op_bin_repr, 8, 0)
