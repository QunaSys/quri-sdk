# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import CNOT, CZ, SWAP, H, QuantumCircuit, X
from quri_parts.circuit.topology import (
    SquareLattice,
    SquareLatticeSWAPInsertionTranspiler,
)


def test_square_lattice_swap_insertion() -> None:
    circuit = QuantumCircuit(9)
    gates = [
        H(0),
        CNOT(0, 8),
        CZ(3, 0),
        SWAP(5, 1),
        X(4),
        CZ(2, 7),
    ]
    circuit.extend(gates)
    square_lattice = SquareLattice(3, 3)
    transpiled = SquareLatticeSWAPInsertionTranspiler(square_lattice)(circuit)

    expect = QuantumCircuit(9)
    expect_gates = [
        H(0),
        SWAP(0, 1),
        SWAP(1, 2),
        SWAP(2, 5),
        CNOT(5, 8),
        SWAP(2, 5),
        SWAP(1, 2),
        SWAP(0, 1),
        CZ(3, 0),
        SWAP(5, 4),
        SWAP(4, 1),
        SWAP(5, 4),
        X(4),
        SWAP(2, 1),
        SWAP(1, 4),
        CZ(4, 7),
        SWAP(1, 4),
        SWAP(2, 1),
    ]
    expect.extend(expect_gates)

    assert transpiled.gates == expect.gates
