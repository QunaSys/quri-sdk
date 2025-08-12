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
from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse as sparse
from typing_extensions import TypeAlias

from . import PAULI_IDENTITY, Operator, PauliLabel, SinglePauli, zero

_sparse_pauli_x = sparse.csc_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_sparse_pauli_y = sparse.csc_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_sparse_pauli_z = sparse.csc_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

_pauli_map = {
    SinglePauli.X: _sparse_pauli_x,
    SinglePauli.Y: _sparse_pauli_y,
    SinglePauli.Z: _sparse_pauli_z,
}


_sparse_matrix_map: dict[str, sparse.spmatrix] = {
    "csc": sparse.csc_matrix,
    "csr": sparse.csr_matrix,
    "bsr": sparse.bsr_matrix,
    "coo": sparse.coo_matrix,
    "dok": sparse.dok_matrix,
    "dia": sparse.dia_matrix,
    "lil": sparse.lil_matrix,
}

SparseMatrixName: TypeAlias = Literal["csc", "csr", "bsr", "coo", "dok", "dia", "lil"]


def _convert_pauli_map_to_other_format(format: SparseMatrixName) -> None:
    if (
        isinstance(_sparse_pauli_x, _sparse_matrix_map[format])
        and isinstance(_sparse_pauli_y, _sparse_matrix_map[format])
        and isinstance(_sparse_pauli_z, _sparse_matrix_map[format])
    ):
        return

    if format == "csc":
        for i in _pauli_map:
            _pauli_map[i] = _pauli_map[i].tocsc()
    elif format == "csr":
        for i in _pauli_map:
            _pauli_map[i] = _pauli_map[i].tocsr()
    elif format == "bsr":
        for i in _pauli_map:
            _pauli_map[i] = _pauli_map[i].tobsr()
    elif format == "coo":
        for i in _pauli_map:
            _pauli_map[i] = _pauli_map[i].tocoo()
    elif format == "dok":
        for i in _pauli_map:
            _pauli_map[i] = _pauli_map[i].todok()
    elif format == "dia":
        for i in _pauli_map:
            _pauli_map[i] = _pauli_map[i].todia()
    elif format == "lil":
        for i in _pauli_map:
            _pauli_map[i] = _pauli_map[i].tolil()
    else:
        assert False, f"format {format} is not supported."


def _convert_pauli_label_to_sparse(
    single_pauli_label: PauliLabel,
    n_qubits: Optional[int] = None,
    format: SparseMatrixName = "csc",
) -> sparse.spmatrix:
    """Convert :class:`~PauliLabel` into scipy sparse matrix."""
    _convert_pauli_map_to_other_format(format)
    if n_qubits is None:
        assert single_pauli_label != PAULI_IDENTITY, (
            "n_qubits needs to be specified for PAULI_IDENTITY"
            " to be converted to matrix."
        )
        n_qubits = max(single_pauli_label.qubit_indices()) + 1

    single_pauli_list = [
        sparse.identity(2, np.complex128, format=format) for _ in range(n_qubits)
    ]

    if single_pauli_label != PAULI_IDENTITY:
        assert n_qubits >= max(single_pauli_label.qubit_indices()) + 1, (
            "The specified number of qubits should not be less then the length"
            " of the pauli operator."
        )
        for bit, pauli in zip(*single_pauli_label.index_and_pauli_id_list):
            single_pauli_list[n_qubits - bit - 1] = _pauli_map[SinglePauli(pauli)]

    return reduce(lambda o1, o2: sparse.kron(o1, o2, format), single_pauli_list)


def _convert_operator_to_sparse(
    operator: Operator, n_qubits: Optional[int] = None, format: SparseMatrixName = "csc"
) -> sparse.spmatrix:
    """Convert :class:`~Operator` into scipy sparse matrix."""
    if operator == zero():
        return (
            _sparse_matrix_map[format](np.zeros((1, 1), dtype=np.complex128))
            if n_qubits is None
            else _sparse_matrix_map[format](
                np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
            )
        )

    if n_qubits is None:
        n_qubits = max(
            [max(op.qubit_indices()) + 1 for op in operator if op != PAULI_IDENTITY]
        )

    return sum(
        [
            coeff * _convert_pauli_label_to_sparse(op, n_qubits, format)
            for op, coeff in operator.items()
        ]
    )


def get_sparse_matrix(
    operator: Union[PauliLabel, Operator],
    n_qubits: Optional[int] = None,
    format: SparseMatrixName = "csc",
) -> sparse.spmatrix:
    """Convert :class:`~PauliLabel` and :class:`~Operator` into scipy sparse
    matrix."""
    if isinstance(operator, PauliLabel):
        return _convert_pauli_label_to_sparse(operator, n_qubits, format)
    elif isinstance(operator, Operator):
        return _convert_operator_to_sparse(operator, n_qubits, format)
    else:
        assert False, "operator should be either a PauliLabel or an Operator object."
