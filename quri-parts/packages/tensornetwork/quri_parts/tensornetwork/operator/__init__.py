# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Sequence
from copy import copy
from functools import reduce
from typing import Any, Literal, Mapping, Optional, Union

import numpy as np
import numpy.typing as npt
import tensornetwork as tn
from tensornetwork import AbstractNode, Edge, Node, split_node
from typing_extensions import TypeAlias

from quri_parts.core.operator import PAULI_IDENTITY, Operator, PauliLabel
from quri_parts.tensornetwork.circuit import TensorNetworkLayer

_PAULI_OPERATOR_DATA_MAP: Sequence[Sequence[Sequence[complex]]] = (
    [[1, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[0, 1j], [-1j, 0]],
    [[1, 0], [0, -1]],
)


class TensorNetworkOperator(TensorNetworkLayer):
    """Tensor network representation of a operators.

    This class subclasses :class:`~TensorNetworkLayer` and provides, in
    addition to input and output edges for the operator, also a list of
    indices that the operator acts on. These indices are defined with
    respect to some quantum state that the operator is intended to act
    on. The intent is to allow for certain optimizations with tensor
    contraction.
    """

    def __init__(
        self,
        index_list: Sequence[int],
        input_edges: Sequence[Edge],
        output_edges: Sequence[Edge],
        container: Union[set[AbstractNode], list[AbstractNode]],
        tensor_map: Sequence[Mapping[int, AbstractNode]],
    ):
        self.index_list = index_list
        super().__init__(input_edges, output_edges, container, tensor_map)

    def copy(self) -> "TensorNetworkOperator":
        """Returns a copy of itself."""
        operator_node_mapping, operator_edge_mapping = tn.copy(
            self._container, conjugate=False
        )
        operator_nodes = {operator_node_mapping[n] for n in self._container}
        operator_input_edges = [operator_edge_mapping[e] for e in self.input_edges]
        operator_output_edges = [operator_edge_mapping[e] for e in self.output_edges]
        tensor_map = [
            {q: operator_node_mapping[n] for q, n in tm.items()}
            for tm in self.tensor_map
        ]

        return TensorNetworkOperator(
            copy(self.index_list),
            operator_input_edges,
            operator_output_edges,
            operator_nodes,
            tensor_map,
        )


_OperatorKey: TypeAlias = Union[PauliLabel, frozenset[tuple[PauliLabel, complex]]]
_operator_cache: dict[_OperatorKey, TensorNetworkOperator] = {}
_mpo_cache: dict[_OperatorKey, TensorNetworkOperator] = {}


def get_observable_data(
    pauli_list: Sequence[Sequence[Sequence[complex]]],
    dim: int = 2,
    n: Optional[int] = None,
) -> npt.NDArray[np.complex128]:
    """Take a list of matrices that represent pauli operators and calculate
    their tensor-product."""

    pauli_list_arrays = [np.array(a, dtype=np.complex128) for a in pauli_list]
    pauli_list_arrays.reverse()
    _obs = reduce(np.kron, pauli_list_arrays)
    if n is None:
        n = len(pauli_list)
    obs_data = np.reshape(_obs, [dim for _ in range(2 * n)])

    return obs_data


def pauli_label_to_array(
    pauli: PauliLabel, index_list: Optional[Collection[int]] = None
) -> npt.NDArray[np.complex128]:
    """Convert a :class:`~PauliLabel` to a numpy array.

    If this function is used to convert a :class:`~PauliLabel` belonging
    to an :class:`~Operator`, then an index_list, describing all of the
    indices acted on by the :class:`~Operator`, should be passed as an
    argument. The returned array will then be padded with identity
    operators where needed.
    """
    if pauli == PAULI_IDENTITY:
        if index_list is None:
            raise ValueError("Cannot convert empty Pauli string to array")
        return get_observable_data(
            [_PAULI_OPERATOR_DATA_MAP[0] for _ in range(len(index_list))]
        )

    if index_list is None:
        this_index_list, pauli_id_list = pauli.index_and_pauli_id_list
        index_list = set(this_index_list)
    else:
        this_index_list, pauli_id_list = pauli.index_and_pauli_id_list
    this_index_list = list(this_index_list)
    pauli_id_list = list(pauli_id_list)
    nq = len(index_list)
    if this_index_list != index_list:
        for q in index_list:
            if q not in this_index_list:
                this_index_list.append(q)
                pauli_id_list.append(0)

    index_pauli_id_map = {q: p for q, p in zip(this_index_list, pauli_id_list)}
    sorted_index_list = sorted(
        this_index_list, reverse=True
    )  # Reverse qubit ordering needed
    sorted_pauli_id_list = [index_pauli_id_map[q] for q in sorted_index_list]
    data_list = [_PAULI_OPERATOR_DATA_MAP[i] for i in sorted_pauli_id_list]

    array = get_observable_data(data_list, dim=2, n=nq)

    return array


def tensor_to_mpo(
    operator: TensorNetworkOperator,
    max_bond_dimension: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    qubits_per_node: int = 1,
) -> TensorNetworkOperator:
    """Perform the singular value decomposition on an operator represented
    using :class:`~TensorNetworkLayer` and return the matrix product operator
    (MPO) as a :class:`~TensorNetworkLayer`

    Args:
        operator: Input operator
        max_bond_dimension: Optional specification of MPO bond-dimension
        max_truncation_error: Optional specification of truncation error tolerance
        qubits_per_node: Number of physical qubits attached to each node, defaults to 1
    Outputs:
        MPO as a :class:`~TensorNetworkOperator`
    """
    assert len(operator._container) == 1

    operator_copy = operator.copy()
    if qubits_per_node == 1:
        edge_groups = list(zip(operator_copy.input_edges, operator_copy.output_edges))
    else:
        raise NotImplementedError()
    right_edges = [
        e
        for contiguous_edges in (operator_copy.input_edges, operator_copy.output_edges)
        for e in contiguous_edges
    ]

    tensor_map = {}
    nodes = set()
    right_node = operator_copy._container.pop()  # Assumes only one node in the operator
    for q, eg in zip(operator_copy.index_list, edge_groups[:-1]):
        left_edges = [e for e in eg]
        right_edges = [e for e in right_edges if e not in left_edges]
        all_edges = left_edges + right_edges
        for e in right_node:
            if e not in all_edges:
                left_edges.append(e)
        left_node, right_node, _ = split_node(
            right_node,
            left_edges,
            right_edges,
            max_singular_values=max_bond_dimension,
            max_truncation_err=max_truncation_err,
        )
        nodes.add(left_node)
        tensor_map[q] = left_node
    tensor_map[operator_copy.index_list[-1]] = right_node
    nodes.add(right_node)

    return TensorNetworkOperator(
        operator_copy.index_list,
        operator_copy.input_edges,
        operator_copy.output_edges,
        nodes,
        [tensor_map],
    )


def operator_to_tensor(
    operator: Union[Operator, PauliLabel],
    convert_to_mpo: bool = True,
    backend: str = "numpy",
    *args: Any,
    **kwargs: Any
) -> TensorNetworkOperator:
    """Convert an :class:`~Operator` or a :class:`~PauliLabel` to a
    :class:`~TensorNetworkOperator`.

    Args:
        operator: Input operator
        qubit_count: Number of qubits of that operator
    Outputs:
        Operator as a :class:`~TensorNetworkOperator`
    """
    op_key: _OperatorKey
    if isinstance(operator, PauliLabel):
        op_key = frozenset({(operator, 1.0)})
    else:
        op_key = frozenset(operator.items())
    if convert_to_mpo:
        cache = _mpo_cache
    else:
        cache = _operator_cache
    if op_key in cache:
        return cache[op_key].copy()

    data: Union[npt.NDArray[np.complex128], Literal[0]]
    if isinstance(operator, PauliLabel):
        qubit_count = len(operator.qubit_indices())
        data = pauli_label_to_array(operator)
        all_indices = set(operator.qubit_indices())
    else:
        assert isinstance(operator, Operator)
        indices_list = [{i for i in p.qubit_indices()} for p in operator.keys()]
        all_indices = set()
        for i in indices_list:
            all_indices.update(i)
        qubit_count = len(all_indices)
        data = sum(
            map(
                lambda p, c: pauli_label_to_array(p, all_indices) * c,
                operator.keys(),
                operator.values(),
            )
        )

    all_indices_list = list(all_indices)
    op = Node(data, backend=backend)
    tensor_map = {q: op for q in all_indices_list}
    tensor = TensorNetworkOperator(
        all_indices_list, op[:qubit_count], op[qubit_count:], {op}, [tensor_map]
    )

    if convert_to_mpo:
        tensor = tensor_to_mpo(tensor, *args, **kwargs)

    if op_key not in cache:
        cache[op_key] = tensor.copy()

    return tensor
