# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections.abc import Sequence
from enum import Enum, auto
from typing import Callable, NamedTuple

from typing_extensions import TypeAlias

from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
)


class EntanglementPatternType(Enum):
    """An enum representing an entanglement pattern for entanglement layers."""

    #: Each qubit is entangled with all the others.
    FULL = auto()
    #: Each qubit (except the last one) is entangled with the next one.
    LINEAR = auto()
    #: Each qubit is entangled with the next one (the first one for the last qubit).
    CIRCULAR = auto()


class RotLayerMakerArg(NamedTuple):
    """Argument type for :class:`~RotLayerMaker`."""

    #: Index of the rotation layer.
    layer_index: int
    #: Index of the qubit.
    qubit_index: int


class EntLayerMakerArg(NamedTuple):
    """Argument type for :class:`~EntLayerMaker`."""

    #: Index of the entanglement layer.
    layer_index: int
    #: Indices of the qubit pair.
    qubit_indices: tuple[int, int]


#: Function to rotate a qubit in a rotation layer.
RotLayerMaker: TypeAlias = Callable[
    [LinearMappedParametricQuantumCircuit, RotLayerMakerArg], None
]
#: Function to entangle a qubit pair in an entanglement layer.
EntLayerMaker: TypeAlias = Callable[
    [LinearMappedParametricQuantumCircuit, EntLayerMakerArg], None
]


class TwoLocal(ImmutableLinearMappedParametricQuantumCircuit):
    """The two-local circuit that consists of rotation layers and entanglement
    layers.

    Examples:
        .. highlight:: python
        .. code-block:: python

            def add_rotation_gate(circuit, arg):
                layer_index, qubit_index = arg
                theta = circuit.add_parameter("theta")
                if layer_index % 2 == 0:
                    circuit.add_ParametricRY_gate(qubit_index, theta)
                else:
                    circuit.add_ParametricRZ_gate(qubit_index, theta)

            def add_entanglement_gate(circuit, arg):
                layer_index, (i, j) = arg
                if layer_index % 2 == 0:
                    circuit.add_CZ_gate(i, j)
                else:
                    circuit.add_CZ_gate(i, j)

            qubit_count = 4
            reps = 2
            entangler_map_seq = build_entangler_map(
                qubit_count, [EntanglementPatternType.LINEAR] * reps
            )

            two_local = TwoLocal(
                qubit_count=qubit_count,
                layer_pattern="rerer",
                rot_layer_maker=add_rotation_gate,
                ent_layer_maker=add_entanglement_gate,
                rotation_indices=range(4),
                entangler_map_seq=entangler_map_seq
            )


    Args:
        qubit_count: Number of qubits.
        layer_pattern: A string specifying layer pattern, which can contain "e"
          (referring an entanglement layer) or "r" (referring a rotation layer).
        rot_layer_maker: Function to rotate a qubit in a rotation layer.
        ent_layer_maker: Function to entangle a qubit pair in an entanglement layer.
        rotation_indices: Qubit indices specifying on which qubits each rotation layer
          acts.
        entangler_map_seq: Qubit index pairs specifying on which qubit pairs each
          entanglement layer acts.

    Raises:
        ValueError: If layer_pattern contains characters other than "e" and "r".
    """

    def __init__(
        self,
        qubit_count: int,
        layer_pattern: str,
        rot_layer_maker: RotLayerMaker,
        ent_layer_maker: EntLayerMaker,
        rotation_indices: Sequence[int],
        entangler_map_seq: Sequence[Sequence[tuple[int, int]]],
    ):
        circuit = _construct_circuit(
            qubit_count,
            layer_pattern,
            rot_layer_maker,
            ent_layer_maker,
            rotation_indices,
            entangler_map_seq,
        )
        super().__init__(circuit)

        #: A string specifying layer pattern, which can contain "e" (referring
        #: an entanglement layer) or "r" (referring a rotation layer).
        self.layer_pattern = layer_pattern
        #: Qubit indices specifying on which qubits each rotation layer acts.
        self.rotation_indices = tuple(rotation_indices)
        #: Qubit index pairs specifying on which qubit pairs each entanglement layer
        #: acts.
        self.entangler_map_seq = tuple(tuple(pairs) for pairs in entangler_map_seq)


def _construct_circuit(
    qubit_count: int,
    layer_pattern: str,
    rot_layer_maker: RotLayerMaker,
    ent_layer_maker: EntLayerMaker,
    rotation_indices: Sequence[int],
    entangler_map_seq: Sequence[Sequence[tuple[int, int]]],
) -> LinearMappedParametricQuantumCircuit:
    if layer_pattern.replace("r", "").replace("e", "") != "":
        raise ValueError("layer_pattern must contain either 'r' or 'e'.")

    circuit = LinearMappedParametricQuantumCircuit(qubit_count)
    ent_layer_index = 0

    for layer_index, p in enumerate(layer_pattern):
        if p == "r":
            _add_rotation_layer(circuit, layer_index, rotation_indices, rot_layer_maker)
        elif p == "e":
            _add_entanglement_layer(
                circuit,
                layer_index,
                entangler_map_seq[ent_layer_index],
                ent_layer_maker,
            )
            ent_layer_index += 1

    return circuit


def _add_rotation_layer(
    circuit: LinearMappedParametricQuantumCircuit,
    layer_index: int,
    rotation_indices: Sequence[int],
    rot_layer_maker: RotLayerMaker,
) -> None:
    for qubit_index in rotation_indices:
        rot_layer_maker(
            circuit,
            RotLayerMakerArg(layer_index, qubit_index),
        )


def _add_entanglement_layer(
    circuit: LinearMappedParametricQuantumCircuit,
    layer_index: int,
    entanglement_pairs: Sequence[tuple[int, int]],
    ent_layer_maker: EntLayerMaker,
) -> None:
    for pair in entanglement_pairs:
        i, j = pair
        ent_layer_maker(
            circuit,
            EntLayerMakerArg(layer_index, (i, j)),
        )


def build_entangler_map(
    qubit_count: int, entanglement_list: Sequence[EntanglementPatternType]
) -> Sequence[Sequence[tuple[int, int]]]:
    """Build an entangler map for one entanglement layer.

    Args:
        qubit_count: Number of qubits.
        entanglement_list: Specification of entanglement patterns in the entanglement
          layer. Each component in the sequence corresponds to a "sub layer" in the
          entanglement layer.

    Returns:
        An entangler map for TwoLocal class.

    Raises:
        ValueError: If the entanglement_list contains an unsupported entanglement
          pattern type.

    Examples:
        >>> build_entangler_map(4, [
            EntanglementPatternType.LINEAR,
            EntanglementPatternType.CIRCULAR,
            EntanglementPatternType.FULL
        ])
        (
            ((1, 2), (0, 1), (2, 3)),
            ((1, 2), (3, 0), (0, 1), (2, 3)),
            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        )
    """
    entangler_map: list[tuple[tuple[int, int], ...]] = []
    for entanglement in entanglement_list:
        entangler_map_layer: list[tuple[int, int]]
        if entanglement == EntanglementPatternType.FULL:
            entangler_map_layer = list(itertools.combinations(range(qubit_count), 2))

        elif entanglement in (
            EntanglementPatternType.LINEAR,
            EntanglementPatternType.CIRCULAR,
        ):
            entangler_map_layer = [(i, i + 1) for i in range(1, qubit_count - 1, 2)]
            for i in range(0, qubit_count - 1, 2):
                entangler_map_layer.append((i, i + 1))
            if entanglement == EntanglementPatternType.CIRCULAR:
                if qubit_count % 2 == 0:
                    entangler_map_layer.insert(
                        (qubit_count - 1) // 2, (qubit_count - 1, 0)
                    )
                else:
                    entangler_map_layer.append((qubit_count - 1, 0))

        else:
            raise ValueError(f"Unsupported EntanglementPatternType: {entanglement}")

        entangler_map.append(tuple(entangler_map_layer))

    return tuple(entangler_map)
