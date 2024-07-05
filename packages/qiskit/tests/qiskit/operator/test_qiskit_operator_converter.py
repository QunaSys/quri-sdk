# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp

from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.qiskit.operator import operator_from_qiskit_op

X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")


def test_parametric_operator_raises() -> None:
    with pytest.raises(
        ValueError, match="Parametric-valued operator is not supported now."
    ):
        op = SparsePauliOp(["II", "XZ"], np.array(ParameterVector("a", 2)))
        operator_from_qiskit_op(op)


def test_operator_from_qiskit_op() -> None:
    qiskit_pauliop = SparsePauliOp(data=Pauli("IIZIYIZ"), coeffs=[2.0])
    expected_psum = Operator(
        {
            pauli_label("Z2 Y4 Z6"): 2,
        }
    )
    assert operator_from_qiskit_op(pauli_operator=qiskit_pauliop) == expected_psum

    qiskit_pauliop = 1j * X ^ Z ^ Y
    expected_psum = Operator(
        {
            pauli_label("X0 Z1 Y2"): 1j,
        }
    )
    assert operator_from_qiskit_op(pauli_operator=qiskit_pauliop) == expected_psum

    qiskit_pauliop = 1 * ((2 * X) ^ (Y + (3 * Z)))
    expected_psum = Operator(
        {
            pauli_label("X0 Y1"): 2,
            pauli_label("X0 Z1"): 6,
        }
    )

    assert operator_from_qiskit_op(pauli_operator=qiskit_pauliop) == expected_psum

    # test: qiskit pauliop -> quriparts Operator (corner case)
    qiskit_pauliop = SparsePauliOp(data=Pauli("III"), coeffs=[2])
    expected_psum = Operator(
        {
            pauli_label(PAULI_IDENTITY): 2,
        }
    )
    assert operator_from_qiskit_op(pauli_operator=qiskit_pauliop) == expected_psum

    qiskit_paulisumop = SparsePauliOp(data=SparsePauliOp(Pauli("IIZI")), coeffs=[2.0])
    qiskit_paulisumop += SparsePauliOp(
        data=SparsePauliOp(Pauli("IIII")), coeffs=[2.0 + 1j]
    )
    qiskit_paulisumop += SparsePauliOp(data=SparsePauliOp(Pauli("XIZI")), coeffs=[0])
    qiskit_paulisumop += SparsePauliOp(
        data=SparsePauliOp(Pauli("XYZI")), coeffs=[200.0 + 1.3j]
    )
    expected_psum = Operator(
        {
            pauli_label("Z2"): 2,
            pauli_label(PAULI_IDENTITY): 2 + 1j,
            pauli_label("X0 Y1 Z2"): 200 + 1.3j,
        }
    )
    assert operator_from_qiskit_op(pauli_operator=qiskit_paulisumop) == expected_psum
