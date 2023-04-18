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

from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    is_hermitian,
    is_ops_close,
    pauli_label,
    truncate,
    zero,
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


def test_constructor(operator: Operator) -> None:
    assert len(operator) == len(PAULI_LABELS)
    for p_label, coef in operator.items():
        assert p_label in PAULI_LABELS
        assert coef in COEFS


def test_add(operator: Operator) -> None:
    other = Operator({pauli_label("X0"): 1.0j, pauli_label("Y0"): 1.0})
    other.constant = 0.5
    sum = operator + other
    assert len(sum) == 5
    assert sum.constant == 1.5
    assert sum[pauli_label("X0")] == 0.1 + 1.0j
    assert sum[pauli_label("Y0")] == 1.0


def test_sub(operator: Operator) -> None:
    other = Operator()
    res = operator - other
    assert operator == res
    res_opposite = other - operator
    for pauli, coef in res_opposite.items():
        assert coef == -1 * operator[pauli]

    other = Operator({PAULI_LABELS[1]: COEFS[1]})
    res = operator - other
    assert len(res) == len(operator) - 1
    assert list(other.keys())[0] not in list(res.keys())
    for pauli, coef in res.items():
        assert coef == operator[pauli]


def test_mul() -> None:
    operator_1 = Operator({pauli_label("X0 Y1 Z2"): 0.1, pauli_label("Z0 X1 Y2"): 0.1j})
    operator_1.constant = 1.0
    operator_2 = Operator({pauli_label("X0 Y1 Z2"): 1.0j, pauli_label("Z0 X1 Y2"): 1.0})
    operator_2.constant = 2.0

    prod = operator_1 * operator_2

    assert len(prod) == 4
    assert prod[PAULI_IDENTITY] == 2.0 + 0.2j
    assert prod[pauli_label("X0 Y1 Z2")] == 0.2 + 1.0j
    assert prod[pauli_label("Z0 X1 Y2")] == 1.0 + 0.2j
    assert prod[pauli_label("Y0 Z1 X2")] == 0.2j


def test_truediv(operator: Operator) -> None:
    with pytest.raises(TypeError):
        operator / Operator({pauli_label("Z0"): 1.0})

    quot = operator / 2
    for p_label, coef in operator.items():
        assert quot[p_label] == coef / 2

    quot = operator / -2.0
    for p_label, coef in operator.items():
        assert quot[p_label] == coef / -2.0

    quot = operator / (1.0 + 1.0j)
    for p_label, coef in operator.items():
        assert quot[p_label] == coef / (1.0 + 1.0j)


def test_copy(operator: Operator) -> None:
    copied = operator.copy()
    assert id(copied) != id(operator)
    assert copied == operator


def test_add_term() -> None:
    operator = Operator()
    operator.add_term(pauli_label({(0, 1)}), 1.0 + 2.0j)
    assert len(operator) == 1
    for p_label, coef in operator.items():
        assert p_label == pauli_label({(0, 1)})
        assert coef == 1.0 + 2.0j


def test_add_duplicated_pauli_label_operator(operator: Operator) -> None:
    """Add an operator COEFS[1]*PAULI_LABELS[0], which means only its
    coefficient is different from the corresponding term in `operator`"""
    operator.add_term(PAULI_LABELS[0], COEFS[1])
    assert len(operator) == len(PAULI_LABELS)
    assert operator[PAULI_LABELS[0]] == COEFS[0] + COEFS[1]


def test_add_constant_twice(operator: Operator) -> None:
    operator.add_term(PAULI_IDENTITY, 2.0j)
    assert operator.constant == COEFS[-1] + 2.0j


def test_add_opposite_sign_term() -> None:
    coef = 1.0 + 2.0j
    operator = Operator()
    operator.add_term(pauli_label({(0, 1)}), coef)
    assert operator.n_terms == 1
    operator.add_term(pauli_label({(0, 1)}), -coef)
    assert operator.n_terms == 0


def test_add_zero_coefficient_term(operator: Operator) -> None:
    operator.add_term(pauli_label({(0, 1)}), 0)
    assert operator.n_terms == len(PAULI_LABELS)


def test_n_terms(operator: Operator) -> None:
    assert operator.n_terms == len(operator)


def test_constant_getter(operator: Operator) -> None:
    assert operator.constant == COEFS[-1]


def test_constant_setter(operator: Operator) -> None:
    new_const = 2.0j
    operator.constant = new_const
    assert operator.constant == new_const


def test_hermitian_conjugated(operator: Operator) -> None:
    op_dag = operator.hermitian_conjugated()
    for p_label, coef in zip(PAULI_LABELS, COEFS):
        assert op_dag[p_label] == coef.conjugate()


def test_is_ops_close() -> None:
    operator_1 = Operator()
    operator_2 = Operator()
    assert is_ops_close(operator_1, zero())
    assert is_ops_close(operator_1, operator_2)

    operator_1.constant = 1.0
    assert not is_ops_close(operator_1, operator_2)

    operator_2.constant = 1.0j
    assert not is_ops_close(operator_1, operator_2)

    operator_2[PAULI_IDENTITY] = 1.0
    assert is_ops_close(operator_1, operator_2)

    operator_1[pauli_label("Z0")] = 2.0
    operator_2[pauli_label("Z0")] = 2.0
    assert is_ops_close(operator_1, operator_2)

    operator_1[pauli_label("X0")] = 1.0 + 0.1j
    operator_2[pauli_label("X0")] = 1.0 - 0.1j
    assert not is_ops_close(operator_1, operator_2)
    operator_2[pauli_label("X0")] = 1.0 + 0.1j
    assert is_ops_close(operator_1, operator_2)

    operator_1[pauli_label("X0 Y1")] = 3.0 + 3.0j
    operator_2[pauli_label("X0 Y1")] = 2.8 + 2.6j
    assert not is_ops_close(operator_1, operator_2)
    assert not is_ops_close(operator_1, operator_2, rtol=1e-1)
    assert is_ops_close(operator_1, operator_2, rtol=2e-1)
    assert not is_ops_close(operator_1, operator_2, rtol=1e-2)

    assert not is_ops_close(zero(), Operator({PAULI_IDENTITY: 1e-4}))
    assert not is_ops_close(zero(), Operator({PAULI_IDENTITY: 1e-4}), rtol=1e-1)
    assert is_ops_close(zero(), Operator({PAULI_IDENTITY: 1e-4}), atol=1e-3)


def test_truncate() -> None:
    operator = Operator()
    assert truncate(operator) == zero()

    operator.constant = 0.01
    assert truncate(operator) == operator
    assert truncate(operator, atol=0.1) == zero()

    operator.constant = 0.01j
    assert truncate(operator) == operator
    assert truncate(operator, atol=0.1) == zero()

    operator[pauli_label("Z0")] = 0.01
    assert truncate(operator) == operator
    assert truncate(operator, atol=0.1) == zero()

    operator[pauli_label("Z1")] = 0.001j
    assert truncate(operator) == operator
    assert truncate(operator, atol=0.005) == Operator(
        {PAULI_IDENTITY: 0.01j, pauli_label("Z0"): 0.01}
    )


def test_is_hermitian() -> None:
    assert is_hermitian(Operator({PAULI_IDENTITY: 1.0}))
    assert not is_hermitian(Operator({PAULI_IDENTITY: 1.0j}))
    assert is_hermitian(Operator({pauli_label("X0 Y1 Z2"): 2.0}))
    assert not is_hermitian(Operator({pauli_label("X0 Y1 Z2"): 2.0 + 0.01j}))
    assert is_hermitian(Operator({pauli_label("X0 Y1 Z2"): 2.0 + 0.01j}), atol=0.1)
