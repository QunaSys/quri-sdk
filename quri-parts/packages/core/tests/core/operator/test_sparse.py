# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce

import numpy as np
import pytest
import scipy.sparse as sparse

from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    get_sparse_matrix,
    pauli_label,
    zero,
)

I = np.eye(2, dtype=np.complex128)  # noqa: E741
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class TestGetSparseMatrix:
    def test_get_sparse_matrix_pauli_label(self) -> None:
        test_pauli_label = pauli_label("X0 X1 Y3 Z4")
        expected_matrix = reduce(np.kron, [Z, Y, I, X, X])
        assert np.allclose(
            get_sparse_matrix(test_pauli_label).toarray(), expected_matrix
        )
        assert isinstance(get_sparse_matrix(test_pauli_label), sparse.csc_matrix)

    def test_get_sparse_matrix_sparse_matrix_type(self) -> None:
        test_pauli_label = pauli_label("X0 X1 Y3 Z4")
        assert isinstance(get_sparse_matrix(test_pauli_label), sparse.csc_matrix)

        assert isinstance(
            get_sparse_matrix(test_pauli_label, format="csr"), sparse.csr_matrix
        )

        assert isinstance(
            get_sparse_matrix(test_pauli_label, format="bsr"), sparse.bsr_matrix
        )

        assert isinstance(
            get_sparse_matrix(test_pauli_label, format="coo"), sparse.coo_matrix
        )

        assert isinstance(
            get_sparse_matrix(test_pauli_label, format="dia"), sparse.dia_matrix
        )

        assert isinstance(
            get_sparse_matrix(test_pauli_label, format="dok"), sparse.dok_matrix
        )

        assert isinstance(
            get_sparse_matrix(test_pauli_label, format="lil"), sparse.lil_matrix
        )

    def test_get_sparse_matrix_nqubit_specified(self) -> None:
        test_pauli_label = pauli_label("X0 X1 Y3 Z4")
        expected_matrix = reduce(np.kron, [I, Z, Y, I, X, X])
        assert np.allclose(
            get_sparse_matrix(test_pauli_label, 6).toarray(), expected_matrix
        )

        with pytest.raises(
            AssertionError,
            match=(
                "The specified number of qubits should not be less then the length"
                " of the pauli operator."
            ),
        ):
            get_sparse_matrix(test_pauli_label, 3)

    def test_get_sparse_matrix_for_operator(self) -> None:
        operator = Operator({pauli_label("X0 Y3"): -3, PAULI_IDENTITY: 8})
        expected_matrix = -3 * reduce(np.kron, [Y, I, I, X]) + 8 * np.eye(16)
        assert np.allclose(get_sparse_matrix(operator).toarray(), expected_matrix)

    def test_get_sparse_matrix_for_zero(self) -> None:
        operator = zero()
        converted = get_sparse_matrix(operator).toarray()
        assert np.allclose(converted, np.zeros((1, 1)))
        assert converted.shape == (1, 1)

        operator = zero()
        converted = get_sparse_matrix(operator, 3).toarray()
        assert np.allclose(converted, np.zeros((8, 8)))
        assert converted.shape == (8, 8)

    def test_get_sparse_matrix_for_identity(self) -> None:
        test_pauli_label = PAULI_IDENTITY
        assert np.allclose(get_sparse_matrix(test_pauli_label, 3).toarray(), np.eye(8))
        with pytest.raises(
            AssertionError,
            match=(
                "n_qubits needs to be specified for PAULI_IDENTITY"
                " to be converted to matrix."
            ),
        ):
            get_sparse_matrix(test_pauli_label)

        operator = Operator({PAULI_IDENTITY: 8})
        assert np.allclose(get_sparse_matrix(operator, 3).toarray(), 8 * np.eye(8))
        with pytest.raises(ValueError):
            get_sparse_matrix(operator)
