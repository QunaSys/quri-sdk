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


class MultiMappedNode(AbstractNode):  # type: ignore
    """This is a convenience class for tensors representing multiple qubits.

    This node must be initialized with an existing node, it then
    facilitates functionality needed for certain MPS or MPO based
    algorithms.
    """

    _input_qubit_index_mapping: Optional[dict[int, int]]
    _output_qubit_index_mapping: Optional[dict[int, int]]

    def __init__(
        self,
        node: Node,
        input_qubit_edge_mapping: Optional[Mapping[int, Edge]] = None,
        output_qubit_edge_mapping: Optional[Mapping[int, Edge]] = None,
        name: Optional[Text] = None,
    ) -> None:
        self.node = node
        self.backend = node.backend

        # The following is needed for tensornetworks internal logic
        for e in self:  # __get_item__ acesses self.edges
            if e.node1 == self.node:
                e.node1 = self
            if e.node2 == self.node:
                e.node2 = self

        self.name = name
        if input_qubit_edge_mapping:
            self._input_qubit_index_mapping = {}
            for qb in input_qubit_edge_mapping:
                if input_qubit_edge_mapping[qb].node1 == self:
                    self._input_qubit_index_mapping[qb] = input_qubit_edge_mapping[
                        qb
                    ].axis1
                if input_qubit_edge_mapping[qb].node2 == self:
                    self._input_qubit_index_mapping[qb] = input_qubit_edge_mapping[
                        qb
                    ].axis2
        else:
            self._input_qubit_index_mapping = None
        if output_qubit_edge_mapping:
            self._output_qubit_index_mapping = {}
            for qb in output_qubit_edge_mapping:
                if output_qubit_edge_mapping[qb].node1 == self:
                    self._output_qubit_index_mapping[qb] = output_qubit_edge_mapping[
                        qb
                    ].axis1
                if output_qubit_edge_mapping[qb].node2 == self:
                    self._output_qubit_index_mapping[qb] = output_qubit_edge_mapping[
                        qb
                    ].axis2
        else:
            self._output_qubit_index_mapping = None
        assert self.input_qubit_edge_mapping or self.output_qubit_edge_mapping
        if self.input_qubit_edge_mapping and self.output_qubit_edge_mapping:
            assert list(self.input_qubit_edge_mapping.keys()) == list(
                self.output_qubit_edge_mapping.keys()
            )
        if self.input_qubit_edge_mapping:
            self.qubit_indices = list(self.input_qubit_edge_mapping.keys())
        elif self.output_qubit_edge_mapping:
            self.qubit_indices = list(self.output_qubit_edge_mapping.keys())

    @property
    def input_qubit_edge_mapping(self) -> Optional[dict[int, Edge]]:
        if self._input_qubit_index_mapping is not None:
            return {
                qb: self[idx] for qb, idx in self._input_qubit_index_mapping.items()
            }
        else:
            return None

    @property
    def output_qubit_edge_mapping(self) -> Optional[dict[int, Edge]]:
        if self._output_qubit_index_mapping is not None:
            return {
                qb: self[idx] for qb, idx in self._output_qubit_index_mapping.items()
            }
        else:
            return None

    @property
    def dtype(self) -> Tensor:
        return self.node.dtype

    def _check_left(self, e: Edge) -> Optional[Edge]:
        """Check to see if `e` connects to a tensor to the left of this one."""
        if (
            self.input_qubit_edge_mapping is not None
            and e in self.input_qubit_edge_mapping.values()
        ):
            return None
        if (
            self.output_qubit_edge_mapping is not None
            and e in self.output_qubit_edge_mapping.values()
        ):
            return None
        if e.node1 != self:
            n = e.node1
        elif e.node2 != self:
            n = e.node2
        else:
            raise ValueError(f"The tensor {self} has a trace dimension")
        if isinstance(n, MultiMappedNode):
            if max(n.qubit_indices) == min(self.qubit_indices) - 1:
                return e
            else:
                return None
        else:
            raise ValueError("Expected `MultiMappedNode` type")

    def _check_right(self, e: Edge) -> Optional[Edge]:
        """Check to see if `e` connects to a tensor to the right of this
        one."""
        if (
            self.input_qubit_edge_mapping is not None
            and e in self.input_qubit_edge_mapping.values()
        ):
            return None
        if (
            self.output_qubit_edge_mapping is not None
            and e in self.output_qubit_edge_mapping.values()
        ):
            return None
        if e.node1 != self:
            n = e.node1
        elif e.node2 != self:
            n = e.node2
        else:
            raise ValueError(f"The tensor {self} has a trace dimension")
        if isinstance(n, MultiMappedNode):
            if min(n.qubit_indices) == max(self.qubit_indices) + 1:
                return e
            else:
                return None
        else:
            raise ValueError("Expected `MultiMappedNode` type")

    @property
    def left_edge(self) -> Optional[Edge]:
        """Assuming the tensor is in MPO form, returns the left edge.

        This method throws an error if there are multiple left edges,
        such as after a contraction of two MPO layers.
        """
        edge = None
        for e in self:
            returned_edge = self._check_left(e)
            if returned_edge:
                if edge is not None:
                    raise RuntimeError("Tensor has multiple left edges")
                edge = returned_edge
        return edge

    @property
    def right_edge(self) -> Optional[Edge]:
        """Assuming the tensor is in MPO form, returns the right edge.

        This method throws an error if there are multiple right edges,
        such as after a contraction of two MPO layers.
        """
        edge = None
        for e in self:
            returned_edge = self._check_right(e)
            if returned_edge:
                if edge is not None:
                    raise RuntimeError("Tensor has multiple right edges")
                edge = returned_edge
        return edge

    @property
    def left_edges(self) -> Sequence[Edge]:
        edges = []
        for e in self:
            returned_edge = self._check_left(e)
            if returned_edge:
                edges.append(returned_edge)
        return edges

    @property
    def right_edges(self) -> Sequence[Edge]:
        edges = []
        for e in self:
            returned_edge = self._check_right(e)
            if returned_edge:
                edges.append(returned_edge)
        return edges

    @property
    def bond_dimension(self) -> Optional[int]:
        """Gives the highest bond dimension of this tensor."""

        mpo_bonds: list[int] = [
            e.dimension for e in (self.left_edge, self.right_edge) if e is not None
        ]
        if mpo_bonds:
            return max(mpo_bonds)
        else:
            return None

    def flatten_edges(self) -> None:
        if len(self.left_edges) > 1:
            tn.flatten_edges(self.left_edges)
        if len(self.right_edges) > 1:
            tn.flatten_edges(self.right_edges)

    def copy(self, conjugate: bool = False) -> "MultiMappedNode":
        """Returns a copy of itself."""
        node_copy = self.node.copy(conjugate)
        if self.input_qubit_edge_mapping:
            input_mapping = {}
            for qb in self.input_qubit_edge_mapping:
                if self.input_qubit_edge_mapping[qb].node1 == self:
                    input_mapping[qb] = node_copy[
                        self.input_qubit_edge_mapping[qb].axis1
                    ]
                elif self.input_qubit_edge_mapping[qb].node2 == self:
                    input_mapping[qb] = node_copy[
                        self.input_qubit_edge_mapping[qb].axis2
                    ]
                else:
                    raise ValueError(
                        f"The node {self} has edges that are not connectd to it"
                    )
        else:
            input_mapping = None
        if self.output_qubit_edge_mapping:
            output_mapping = {}
            for qb in self.output_qubit_edge_mapping:
                if self.output_qubit_edge_mapping[qb].node1 == self:
                    output_mapping[qb] = node_copy[
                        self.output_qubit_edge_mapping[qb].axis1
                    ]
                elif self.output_qubit_edge_mapping[qb].node2 == self:
                    output_mapping[qb] = node_copy[
                        self.output_qubit_edge_mapping[qb].axis2
                    ]
                else:
                    raise ValueError(
                        f"The node {self} has edges that are not connectd to it"
                    )
        else:
            output_mapping = None
        mapped_node = MultiMappedNode(
            node_copy,
            input_mapping,
            output_mapping,
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
    def _tensor(self) -> Tensor:
        return self.node.tensor

    @_tensor.setter
    def _tensor(self, tensor: Tensor) -> None:
        self.node.tensor = tensor

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, tensor: Tensor) -> None:
        self._tensor = tensor

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

    def __repr__(self) -> Text:
        edges = self.get_all_edges()
        return (
            f"{self.__class__.__name__}\n(\n"
            f"name : {self.name!r},"
            f"\ntensor : \n{self.tensor!r},"
            f"\nedges : \n{edges!r} \n)"
        )


class MappedNode(MultiMappedNode):
    """This is a convenience class for single tensors in an MPS or MPO.

    This node must be initialized with an existing node, it then
    facilitates functionality needed for certain MPS or MPO based
    algorithms.
    """

    def __init__(
        self,
        node: Node,
        qubit_index: int,
        input_edge_index: Optional[int] = None,
        output_edge_index: Optional[int] = None,
        name: Optional[Text] = None,
    ) -> None:
        if input_edge_index is None:
            input_mapping = None
        else:
            input_mapping = {qubit_index: node[input_edge_index]}
        if output_edge_index is None:
            output_mapping = None
        else:
            output_mapping = {qubit_index: node[output_edge_index]}
        MultiMappedNode.__init__(self, node, input_mapping, output_mapping, name=name)
        self.qubit_index = qubit_index
        self.input_edge_index = input_edge_index
        self.output_edge_index = output_edge_index
        if input_edge_index is not None:
            assert self[input_edge_index] in node.edges
            self._input_edge = self[input_edge_index]
        else:
            self._input_edge = None
        if output_edge_index is not None:
            assert self[output_edge_index] in node.edges
            self._output_edge = self[output_edge_index]
        else:
            self._output_edge = None

    @property
    def input_edge(self) -> Edge:
        if self._input_edge:
            return self._input_edge
        else:
            raise ValueError(f"Mappednode {self} does not have an input edge.")

    @property
    def output_edge(self) -> Edge:
        if self._output_edge:
            return self._output_edge
        else:
            raise ValueError(f"Mappednode {self} does not have an output edge.")

    def copy(self, conjugate: bool = False) -> "MappedNode":
        """Returns a copy of itself."""
        node_copy = self.node.copy(conjugate)
        mapped_node = MappedNode(
            node_copy,
            self.qubit_index,
            self.input_edge_index,
            self.output_edge_index,
            self.name,
        )
        return mapped_node


class TensorNetworkState(NodeCollection):  # type: ignore
    """Tensor network representation of a quantum state.

    This class subclasses :class:`~NodeCollection` and provides output
    edges for the state, each of which represents a qubit
    """

    def __init__(
        self,
        edges: Sequence[Edge],
        container: Union[set[AbstractNode], list[AbstractNode]],
        layer_tensor_map: Sequence[Mapping[int, Union[AbstractNode]]],
    ):
        self.edges = edges
        self.layer_tensor_map = layer_tensor_map
        super().__init__(container)

    def with_gates_applied(self, circuit: TensorNetworkLayer) -> "TensorNetworkState":
        """Returns a new :class:`~TensorNetworkState` with the given
        :class:`~TensorNetworkCircuit` applied."""
        circuit = circuit.copy()
        state = self.copy()

        for e, f in zip(state.edges, circuit.input_edges):
            e ^ f

        node_set = state._container.union(circuit._container)
        tensor_map = list(state.layer_tensor_map) + list(circuit.layer_tensor_map)
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
            for mapping in self.layer_tensor_map
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
        mapped_node = MappedNode(node, q, None, 0, name=f"|0> q={q}")
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
