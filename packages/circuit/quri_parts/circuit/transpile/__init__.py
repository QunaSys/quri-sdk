# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

from .clifford_approximation import CliffordApproximationTranspiler
from .gate_kind_decomposer import (
    CZ2CNOTHTranspiler,
    H2RZSqrtXTranspiler,
    RX2RZSqrtXTranspiler,
    RY2RZSqrtXTranspiler,
    S2RZTranspiler,
    Sdag2RZTranspiler,
    SqrtX2RZHTranspiler,
    SqrtXdag2RZSqrtXTranspiler,
    SqrtY2RZSqrtXTranspiler,
    SqrtYdag2RZSqrtXTranspiler,
    SWAP2CNOTTranspiler,
    T2RZTranspiler,
    Tdag2RZTranspiler,
    U1ToRZTranspiler,
    U2ToRZSqrtXTranspiler,
    U3ToRZSqrtXTranspiler,
    X2HZTranspiler,
    X2SqrtXTranspiler,
    Y2RZXTranspiler,
    Z2HXTranspiler,
    Z2RZTranspiler,
)
from .identity_insertion import IdentityInsertionTranspiler
from .multi_pauli_decomposer import (
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
)
from .qubit_remapping import QubitRemappingTranspiler
from .transpiler import (
    CircuitTranspiler,
    CircuitTranspilerProtocol,
    GateDecomposer,
    GateKindDecomposer,
    ParallelDecomposer,
    SequentialTranspiler,
)

#: CircuitTranspiler to transpile a QuntumCircuit into another
#: QuantumCircuit containing only X, SqrtX, CNOT, and RZ.
RZSetTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [
        ParallelDecomposer(
            [
                CZ2CNOTHTranspiler(),
                PauliDecomposeTranspiler(),
                PauliRotationDecomposeTranspiler(),
            ]
        ),
        ParallelDecomposer(
            [
                Y2RZXTranspiler(),
                Z2RZTranspiler(),
                H2RZSqrtXTranspiler(),
                SqrtXdag2RZSqrtXTranspiler(),
                SqrtY2RZSqrtXTranspiler(),
                SqrtYdag2RZSqrtXTranspiler(),
                S2RZTranspiler(),
                Sdag2RZTranspiler(),
                T2RZTranspiler(),
                Tdag2RZTranspiler(),
                RX2RZSqrtXTranspiler(),
                RY2RZSqrtXTranspiler(),
                U1ToRZTranspiler(),
                U2ToRZSqrtXTranspiler(),
                U3ToRZSqrtXTranspiler(),
            ]
        ),
    ]
)


__all__ = [
    "CircuitTranspiler",
    "CircuitTranspilerProtocol",
    "SequentialTranspiler",
    "GateDecomposer",
    "GateKindDecomposer",
    "ParallelDecomposer",
    "CliffordApproximationTranspiler",
    "IdentityInsertionTranspiler",
    "CZ2CNOTHTranspiler",
    "H2RZSqrtXTranspiler",
    "PauliDecomposeTranspiler",
    "PauliRotationDecomposeTranspiler",
    "QubitRemappingTranspiler",
    "RX2RZSqrtXTranspiler",
    "RY2RZSqrtXTranspiler",
    "RZSetTranspiler",
    "S2RZTranspiler",
    "Sdag2RZTranspiler",
    "SqrtX2RZHTranspiler",
    "SqrtXdag2RZSqrtXTranspiler",
    "SqrtY2RZSqrtXTranspiler",
    "SqrtYdag2RZSqrtXTranspiler",
    "SWAP2CNOTTranspiler",
    "T2RZTranspiler",
    "Tdag2RZTranspiler",
    "U1ToRZTranspiler",
    "U2ToRZSqrtXTranspiler",
    "U3ToRZSqrtXTranspiler",
    "X2HZTranspiler",
    "X2SqrtXTranspiler",
    "Y2RZXTranspiler",
    "Z2HXTranspiler",
    "Z2RZTranspiler",
]
