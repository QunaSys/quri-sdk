# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Mapping, Sequence
from typing import Callable, Optional

from typing_extensions import TypeAlias

from quri_parts.circuit import QuantumGate, gate_names, gates
from quri_parts.circuit.transpile import GateDecomposer

Coordinate: TypeAlias = tuple[int, int]
CoordToQubitMap: TypeAlias = Mapping[Coordinate, int]
QubitToCoordMap: TypeAlias = Mapping[int, Coordinate]


class SquareLattice:
    """Represents qubits connected in a rectangular lattice with no missing
    pieces, and searches for the shortest path between any two qubits.

    Args:
        xsize: Size of lattice in x direction.
        ysize: Size of lattice in y direction.
        coord_to_qubit: Mapping from coordinates to qubit indices. If omitted,
            it is automatically assigned from the origin with x changing first.
    """

    def __init__(
        self, xsize: int, ysize: int, qubit_to_coord: Optional[QubitToCoordMap] = None
    ):
        if qubit_to_coord is not None:
            self._qubit_to_coord = qubit_to_coord
        else:
            self._qubit_to_coord = self._create_aligned_map(xsize, ysize)
        self._coord_to_qubit: CoordToQubitMap = {}
        for k, v in self._qubit_to_coord.items():
            if v in self._coord_to_qubit:
                raise ValueError(
                    f"The same coordinates were specified multiple times: {v}."
                )
            self._coord_to_qubit[v] = k
        self._xsize, self._ysize = xsize, ysize

    @property
    def xsize(self) -> int:
        return self._xsize

    @property
    def ysize(self) -> int:
        return self._ysize

    @property
    def qubits(self) -> Collection[int]:
        return self._qubit_to_coord.keys()

    def get_qubit(self, coord: Coordinate) -> int:
        """Returns qubit index corresponding to the given coordinate."""
        return self._coord_to_qubit[coord]

    def get_coord(self, qubit: int) -> Coordinate:
        """Returns coordinate corresponding to the given qubit index."""
        return self._qubit_to_coord[qubit]

    def is_adjacent(self, qubit_a: int, qubit_b: int) -> bool:
        """Check if the given qubits are directory connected to each other on
        the lattice."""
        if qubit_a == qubit_b:
            raise ValueError("Connections between the same qubit indices were tested.")
        xa, ya = self.get_coord(qubit_a)
        xb, yb = self.get_coord(qubit_b)
        return abs(xa - xb) + abs(ya - yb) == 1

    def qubit_path(self, qubit_a: int, qubit_b: int) -> Sequence[int]:
        """Returns sequence of qubits on the shortest path between the
        coordinates of two qubits.

        Args:
            qubit_a: Qubit index at the starting point.
            qubit_b: Qubit index at the end point.
        """
        path = []
        sx, sy = self.get_coord(qubit_a)
        ex, ey = self.get_coord(qubit_b)
        xstep = 1 if ex > sx else -1
        ystep = 1 if ey > sy else -1
        for x in range(sx, ex, xstep):
            path.append(self.get_qubit((x, sy)))
        for y in range(sy, ey + ystep, ystep):
            path.append(self.get_qubit((ex, y)))
        return path

    def _create_aligned_map(self, xsize: int, ysize: int) -> QubitToCoordMap:
        """Create a :class:`QubitToCoordMap` such that the qubit indices
        correspond in order from the origin. The x coordinate changes first.

        Args:
            xsize: Size of lattice in x direction.
            ysize: Size of lattice in y direction.
        """
        i = 0
        ret = {}
        for y in range(ysize):
            for x in range(xsize):
                ret[i] = (x, y)
                i += 1
        return ret


class SquareLatticeSWAPInsertionTranspiler(GateDecomposer):
    """When the connection of qubits is expressed in :class:`SquareLattice`,
    insert SWAPs as appropriate before and after for gates applied between two
    qubits that are not connected to each other."""

    def __init__(self, square_lattice: SquareLattice):
        self._square_lattice = square_lattice

    def is_target_gate(self, gate: QuantumGate) -> bool:
        return len(gate.control_indices) + len(gate.target_indices) > 1

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        if gate.name not in {gate_names.CNOT, gate_names.SWAP, gate_names.CZ}:
            raise ValueError(f"Unsupported multi qubit gate: {gate.name}")

        qubit_a, qubit_b = tuple(gate.control_indices) + tuple(gate.target_indices)

        if self._square_lattice.is_adjacent(qubit_a, qubit_b):
            return [gate]

        before: list[QuantumGate] = []
        path = self._square_lattice.qubit_path(qubit_a, qubit_b)[:-1]
        dst = path[-1]
        for a, b in zip(path, path[1:]):
            before.append(gates.SWAP(a, b))
        after = reversed(before)

        fs: Mapping[str, Callable[[int, int], QuantumGate]] = {
            gate_names.CNOT: gates.CNOT,
            gate_names.SWAP: gates.SWAP,
            gate_names.CZ: gates.CZ,
        }
        own = fs[gate.name](dst, qubit_b)

        return tuple(before) + (own,) + tuple(after)
