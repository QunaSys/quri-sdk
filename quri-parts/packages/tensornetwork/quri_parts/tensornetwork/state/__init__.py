# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Mapping, Optional, Sequence, Text, Union

import numpy as np
import tensornetwork as tn
from h5py import Group
from tensornetwork import AbstractNode, Edge, Node, NodeCollection, Tensor

from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.core.state import CircuitQuantumState, QuantumStateVector
from quri_parts.tensornetwork.circuit import (
    TensorNetworkLayer,
    TensorNetworkTranspiler,
    convert_circuit,
)


class MappedNode(AbstractNode):  # type: ignore
    """This is a convenience class for single-qubits."""

    def __init__(
        self,
        node: Node,
        qubit_index: int,
        qubit_edge_index: int,
        name: Optional[Text] = None,
    ) -> None:
        self.node = node
        self.backend = node.backend
        self.qubit_index = qubit_index
        self.qubit_edge_index = qubit_edge_index  # This may not be 0 in an MPS
        for e in self:
            if e.node1 == self.node:
                e.node1 = self
            if e.node2 == self.node:
                e.node2 = self
        if name is not None:
            self.name = name

    @property
    def dtype(self) -> Tensor:
        return self.node.dtype

    @property
    def qubit_edge(self) -> Edge:
        return self[self.qubit_edge_index]

    def copy(self, conjugate: bool = False) -> "MappedNode":
        """Returns a copy of itself."""
        node_copy = self.node.copy(conjugate)
        mapped_node = MappedNode(
            node_copy,
            self.qubit_index,
            self.qubit_edge_index,
            self.name,
        )
        return mapped_node

    @property
    def edges(self) -> Any:
        return self.node.edges

    @edges.setter
    def edges(self, edges: List["Edge"]) -> None:
        self.node.edges = edges

    @property
    def name(self) -> Any:
        return self.node.name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("Node name should be str type")
        self.node.name = name

    @property
    def axis_names(self) -> Any:
        return self.node.axis_names

    @axis_names.setter
    def axis_names(self, axis_names: List[Text]) -> None:
        self.node.axis_names = axis_names

    def disable(self) -> None:
        if self.node.is_disabled:
            raise ValueError("Node {} is already disabled".format(self.name))
        self.node.is_disabled = True

    def get_tensor(self) -> Tensor:
        return self.node.get_tensor()

    def set_tensor(self, tensor: Tensor) -> None:
        self.node.set_tensor(tensor)

    @property
    def shape(self) -> Any:
        return self.node.shape

    @property
    def tensor(self) -> Tensor:
        return self.node.tensor

    @tensor.setter
    def tensor(self, tensor: Tensor) -> None:
        self.node.tensor = tensor

    @classmethod
    def _load_node(cls, _: Group) -> "AbstractNode":
        """load a node based on hdf5 data.

        Args:
          node_data: h5py group that contains the serialized node data

        Returns:
          The loaded node.
        """
        raise NotImplementedError("Loading nodes is not supported for MappedNode")

    def _save_node(self, _: Group) -> None:
        """Abstract method to enable saving nodes to hdf5. Only serializing
        common properties is implemented. Should be overwritten by subclasses.

        Args:
          node_group: h5py group where data is saved
        """
        raise NotImplementedError("Saving nodes is not supported for MappedNode")

    def to_serial_dict(self) -> None:
        """Return a serializable dict representing the node.

        Returns: A dict object.
        """
        raise NotImplementedError("Serializing nodes is not supported for MappedNode")

    @classmethod
    def from_serial_dict(cls, _: Any) -> None:
        """Return a node given a serialized dict representing it.

        Args:
          serial_dict: A python dict representing a serialized node.

        Returns:
          A node.
        """
        raise NotImplementedError("Serializing nodes is not supported for MappedNode")


class TensorNetworkState(NodeCollection):  # type: ignore
    """Tensor network representation of a quantum state.

    This class subclasses :class:`~NodeCollection` and provides output
    edges for the state, each of which represents a qubit
    """

    def __init__(
        self,
        edges: Sequence[Edge],
        container: Union[set[AbstractNode], list[AbstractNode]],
        tensor_map: Sequence[Mapping[int, Union[AbstractNode]]],
    ):
        self.edges = edges
        self.tensor_map = tensor_map
        super().__init__(container)

    def with_gates_applied(self, circuit: TensorNetworkLayer) -> "TensorNetworkState":
        """Returns a new :class:`~TensorNetworkState` with the given
        :class:`~TensorNetworkCircuit` applied."""
        circuit = circuit.copy()
        state = self.copy()

        for e, f in zip(state.edges, circuit.input_edges):
            e ^ f

        node_set = state._container.union(circuit._container)
        tensor_map = list(state.tensor_map) + list(circuit.tensor_map)
        return TensorNetworkState(circuit.output_edges, node_set, tensor_map)

    def copy(self, conjugate: bool = False) -> "TensorNetworkState":
        """Returns a copy of itself."""
        state_node_mapping, state_edge_mapping = tn.copy(
            self._container, conjugate=conjugate
        )
        state_nodes = {state_node_mapping[n] for n in self._container}
        state_edges = [state_edge_mapping[e] for e in self.edges]
        tensor_map = [
            {q: state_node_mapping[n] for q, n in mapping.items()}
            for mapping in self.tensor_map
        ]

        return TensorNetworkState(state_edges, state_nodes, tensor_map)

    def conjugate(self) -> "TensorNetworkState":
        """Returns a conjugated copy of itself."""

        return self.copy(conjugate=True)

    def contract(self, method: str = "greedy") -> "TensorNetworkState":
        """Returns a copy of self after contracting internal tensor network."""
        copy = self.copy()
        if method == "greedy":
            node = tn.contractors.greedy(copy._container, output_edge_order=copy.edges)
        else:
            raise NotImplementedError(
                "The requested contraction algorithms is not available"
            )
        tensor_map = {q: node for q in range(len(copy.edges))}

        return TensorNetworkState(copy.edges, {node}, [tensor_map])


def get_zero_state(qubit_count: int, backend: str = "numpy") -> TensorNetworkState:
    """Returns the zero state for the given number of qubits."""
    qubits: list[MappedNode] = []
    zero_state_edges: list[Edge] = []
    tensor_map: dict[int, MappedNode] = {}
    for q in range(qubit_count):
        node = Node(np.array([1.0, 0.0], dtype=np.complex128), backend=backend)
        mapped_node = MappedNode(node, q, 0, name=f"|0> q={q}")
        qubits.append(mapped_node)
        zero_state_edges.append(mapped_node[0])
        tensor_map[q] = mapped_node
    return TensorNetworkState(zero_state_edges, qubits, [tensor_map])


def convert_state(
    state: CircuitQuantumState,
    transpiler: Optional[CircuitTranspiler] = TensorNetworkTranspiler(),
    backend: str = "numpy",
) -> TensorNetworkState:
    qubit_count = state.qubit_count
    zero_state = get_zero_state(qubit_count, backend=backend)
    state_circuit = convert_circuit(
        state.circuit, transpiler=transpiler, backend=backend
    )
    tn_state = zero_state.with_gates_applied(state_circuit)
    return tn_state


def quantum_state_vector_from_tensor_network_state(
    state: TensorNetworkState,
) -> QuantumStateVector:
    """Convert TensorNetworkState to QuantumStateVector that only contains the
    state vector and no quantum circuit."""
    state = state.contract()
    qubit_count = len(state.edges)

    tensor = np.reshape(
        state._container.pop().tensor,
        (2**qubit_count),
        order="F",
    )

    return QuantumStateVector(qubit_count, tensor)
