# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Mapping, Optional, Sequence, Union

import tensornetwork as tn
from tensornetwork import AbstractNode, Edge, NodeCollection

from quri_parts.circuit import ImmutableQuantumCircuit, gate_names
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
)
from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)
from quri_parts.tensornetwork.circuit import gates
from quri_parts.tensornetwork.circuit.gates import (
    SingleQubitGate,
    SingleQubitPauliRotationGate,
    SingleQubitRotationGate,
    TensorNetworkQuantumGate,
    ThreeQubitGate,
    TwoQubitGate,
)

_single_qubit_gate_tensornetwork: Mapping[
    SingleQubitGateNameType, type[SingleQubitGate]
] = {
    gate_names.Identity: gates.I,
    gate_names.X: gates.X,
    gate_names.Y: gates.Y,
    gate_names.Z: gates.Z,
    gate_names.H: gates.H,
    gate_names.S: gates.S,
    gate_names.Sdag: gates.Sdag,
    gate_names.SqrtX: gates.SqrtX,
    gate_names.SqrtXdag: gates.SqrtXdag,
    gate_names.SqrtY: gates.SqrtY,
    gate_names.SqrtYdag: gates.SqrtYdag,
    gate_names.T: gates.T,
    gate_names.Tdag: gates.Tdag,
}

_single_qubit_pauli_rotation_gate_tensornetwork: Mapping[
    SingleQubitGateNameType, type[SingleQubitPauliRotationGate]
] = {
    gate_names.RX: gates.Rx,
    gate_names.RY: gates.Ry,
    gate_names.RZ: gates.Rz,
}

_single_qubit_rotation_gate_tensornetwork: Mapping[
    SingleQubitGateNameType, type[SingleQubitRotationGate]
] = {
    gate_names.U1: gates.U1,
    gate_names.U2: gates.U2,
    gate_names.U3: gates.U3,
}

_two_qubit_gate_tensornetwork: Mapping[TwoQubitGateNameType, type[TwoQubitGate]] = {
    gate_names.CNOT: gates.CNOT,
    gate_names.CZ: gates.CZ,
    gate_names.SWAP: gates.SWAP,
}

_three_qubit_gate_tensornetwork: Mapping[
    ThreeQubitGateNameType, type[ThreeQubitGate]
] = {
    gate_names.TOFFOLI: gates.Toffoli,
}

#: CircuitTranspiler to convert a circit configuration suitable for tensornetwork.
TensorNetworkTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
)


class TensorNetworkLayer(NodeCollection):  # type: ignore
    """Tensor network representation of a quantum circuit and operators.

    This class subclasses :class:`~NodeCollection` and provides input
    and output edges for the circuit/operator, each of which represents
    a qubit.
    """

    def __init__(
        self,
        input_edges: Sequence[Edge],
        output_edges: Sequence[Edge],
        container: Union[set[AbstractNode], list[AbstractNode]],
        tensor_map: Sequence[Mapping[int, TensorNetworkQuantumGate]],
    ):
        self.input_edges = input_edges
        self.output_edges = output_edges
        self.tensor_map = tensor_map
        super().__init__(container)

    def copy(self) -> "TensorNetworkLayer":
        """Returns a copy of itself."""
        circuit_node_mapping, circuit_edge_mapping = tn.copy(
            self._container, conjugate=False
        )
        circuit_nodes = {circuit_node_mapping[n] for n in self._container}
        tensor_map = [
            {q: circuit_node_mapping[n] for q, n in mapping.items()}
            for mapping in self.tensor_map
        ]
        circuit_input_edges = [circuit_edge_mapping[e] for e in self.input_edges]
        circuit_output_edges = [circuit_edge_mapping[e] for e in self.output_edges]

        return TensorNetworkLayer(
            circuit_input_edges, circuit_output_edges, circuit_nodes, tensor_map
        )


def connect_gate(
    node: TensorNetworkQuantumGate,
    qubits: Sequence[int],
    qubit_count: int,
    depth: list[int],
    tensor_map: list[dict[int, TensorNetworkQuantumGate]],
    in_out_map: Sequence[dict[str, Optional[Edge]]],
) -> None:
    max_depth = max(depth[q] for q in qubits)
    for q in qubits:
        depth[q] = max_depth
    for i, q in enumerate(qubits):
        if in_out_map[q]["in"]:
            in_out_map[q]["out"] ^ node[i]
            in_out_map[q]["out"] = node[qubit_count + i]
        else:
            assert in_out_map[q]["out"] is None
            in_out_map[q]["in"] = node[i]
            in_out_map[q]["out"] = node[qubit_count + i]
        if depth[q] >= len(tensor_map):
            tensor_map.append({})
        if tensor_map[depth[q]].get(q) is None:
            tensor_map[depth[q]][q] = node
        depth[q] += 1


def add_disconnected_qubits(
    in_out_map: Sequence[dict[str, Optional[Edge]]], node_collection: NodeCollection
) -> None:
    for q, m in enumerate(in_out_map):
        if m["in"] is None or m["out"] is None:
            assert m["in"] is None and m["out"] is None
            node = gates.I([q])
            m["in"] = node[0]
            m["out"] = node[1]
            node_collection.add(node)


def convert_circuit(
    circuit: ImmutableQuantumCircuit,
    transpiler: Optional[CircuitTranspiler] = TensorNetworkTranspiler(),
    backend: str = "numpy",
) -> TensorNetworkLayer:
    """Convert an :class:`~ImmutableQuantumCircuit` to a tensornetwork
    NodeCollection.

    Args:
        circuit: the quantum circuit to convert to a node collection
        transpiler: optional transpiler to use
    """

    if transpiler is not None:
        circuit = transpiler(circuit)

    qubit_count = circuit.qubit_count
    in_out_map: Sequence[dict[str, Optional[Edge]]] = [
        {"in": None, "out": None} for _ in range(qubit_count)
    ]
    depth: list[int] = [0 for _ in range(qubit_count)]
    tensor_map: list[dict[int, TensorNetworkQuantumGate]] = [
        {} for _ in range(circuit.depth)
    ]

    node_collection = set()
    for gate in circuit.gates:
        if not is_gate_name(gate.name):
            raise ValueError(f"Unknown gate name: {gate.name}")

        if gate.name == "Identity":
            continue

        if is_single_qubit_gate_name(gate.name):
            if gate.name in _single_qubit_gate_tensornetwork:
                node = _single_qubit_gate_tensornetwork[gate.name](
                    gate.target_indices,  # type: ignore[arg-type]
                    backend=backend,  # type: ignore[call-arg]
                )
                node_collection.add(node)
            elif gate.name in _single_qubit_pauli_rotation_gate_tensornetwork:
                if len(gate.params) == 1:
                    node = _single_qubit_pauli_rotation_gate_tensornetwork[gate.name](
                        gate.params[0],
                        gate.target_indices,
                        backend=backend,  # type: ignore[call-arg]
                    )
                    node_collection.add(node)
                else:
                    raise ValueError("Invalid number of parameters.")
            elif gate.name in _single_qubit_rotation_gate_tensornetwork:
                node = _single_qubit_pauli_rotation_gate_tensornetwork[gate.name](
                    gate.params,  # type: ignore[arg-type]
                    gate.target_indices,
                    backend=backend,  # type: ignore[call-arg]
                )
                node_collection.add(node)
            else:
                raise ValueError(f"{gate.name} gate is not supported.")
            connect_gate(node, gate.target_indices, 1, depth, tensor_map, in_out_map)
        elif is_two_qubit_gate_name(gate.name):
            if gate.name not in _two_qubit_gate_tensornetwork:
                raise ValueError(f"{gate.name} gate is not supported.")
            if gate.name == "SWAP":
                indices = gate.target_indices
            else:
                indices = list(gate.control_indices) + list(gate.target_indices)
            node = _two_qubit_gate_tensornetwork[gate.name](
                indices,  # type: ignore[arg-type]
                backend=backend,  # type: ignore[call-arg]
            )
            node_collection.add(node)
            connect_gate(node, indices, 2, depth, tensor_map, in_out_map)
        elif is_three_qubit_gate_name(gate.name):
            if gate.name not in _three_qubit_gate_tensornetwork:
                raise ValueError(f"{gate.name} gate is not supported.")
            indices = list(gate.control_indices) + list(gate.target_indices)
            node = _three_qubit_gate_tensornetwork[gate.name](
                indices,  # type: ignore[arg-type]
                backend=backend,  # type: ignore[call-arg]
            )
            node_collection.add(node)
            connect_gate(node, indices, 3, depth, tensor_map, in_out_map)
        else:
            raise ValueError(f"Unknown gate name: {gate.name}")

    add_disconnected_qubits(in_out_map, node_collection)

    input_edges = [m["in"] for m in in_out_map]
    output_edges = [m["out"] for m in in_out_map]

    return TensorNetworkLayer(input_edges, output_edges, node_collection, tensor_map)
