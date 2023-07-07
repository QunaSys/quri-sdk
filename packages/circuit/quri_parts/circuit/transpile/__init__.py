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
from .fuse import (
    FuseRotationTranspiler,
    NormalizeRotationTranspiler,
    RX2NamedTranspiler,
    RY2NamedTranspiler,
    RZ2NamedTranspiler,
)
from .gate_kind_decomposer import (
    CNOT2CZHTranspiler,
    CZ2CNOTHTranspiler,
    CZ2RXRYCNOTTranspiler,
    H2RXRYTranspiler,
    H2RZSqrtXTranspiler,
    Identity2RZTranspiler,
    RX2RZSqrtXTranspiler,
    RY2RZSqrtXTranspiler,
    S2RZTranspiler,
    Sdag2RZTranspiler,
    SqrtX2RXTranspiler,
    SqrtX2RZHTranspiler,
    SqrtXdag2RXTranspiler,
    SqrtXdag2RZSqrtXTranspiler,
    SqrtY2RYTranspiler,
    SqrtY2RZSqrtXTranspiler,
    SqrtYdag2RYTranspiler,
    SqrtYdag2RZSqrtXTranspiler,
    SWAP2CNOTTranspiler,
    T2RZTranspiler,
    Tdag2RZTranspiler,
    TOFFOLI2HTTdagCNOTTranspiler,
    U1ToRZTranspiler,
    U2ToRXRZTranspiler,
    U2ToRZSqrtXTranspiler,
    U3ToRXRZTranspiler,
    U3ToRZSqrtXTranspiler,
    X2HZTranspiler,
    X2RXTranspiler,
    X2SqrtXTranspiler,
    Y2RYTranspiler,
    Y2RZXTranspiler,
    Z2HXTranspiler,
    Z2RZTranspiler,
)
from .identity_manipulation import (
    IdentityEliminationTranspiler,
    IdentityInsertionTranspiler,
)
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
from .unitary_matrix_decomposer import (
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    TwoQubitUnitaryMatrixKAKTranspiler,
    su2_decompose,
    su4_decompose,
)

#: CircuitTranspiler to transpile a QuntumCircuit into another
#: QuantumCircuit containing only X, SqrtX, CNOT, and RZ.
#: (UnitaryMatrix gate for 3 or more qubits are not decomposed.)
RZSetTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [
        SingleQubitUnitaryMatrix2RYRZTranspiler(),
        TwoQubitUnitaryMatrixKAKTranspiler(),
        ParallelDecomposer(
            [
                CZ2CNOTHTranspiler(),
                PauliDecomposeTranspiler(),
                PauliRotationDecomposeTranspiler(),
                TOFFOLI2HTTdagCNOTTranspiler(),
            ]
        ),
        ParallelDecomposer(
            [
                Identity2RZTranspiler(),
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
                SWAP2CNOTTranspiler(),
            ]
        ),
        FuseRotationTranspiler(),
    ]
)


#: CircuitTranspiler to transpile a QuntumCircuit into another
#: QuantumCircuit containing only RX, RY, RZ, and CNOT.
#: (UnitaryMatrix gate for 3 or more qubits are not decomposed.)
RotationSetTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [
        SingleQubitUnitaryMatrix2RYRZTranspiler(),
        TwoQubitUnitaryMatrixKAKTranspiler(),
        ParallelDecomposer(
            [
                PauliDecomposeTranspiler(),
                PauliRotationDecomposeTranspiler(),
                TOFFOLI2HTTdagCNOTTranspiler(),
            ]
        ),
        ParallelDecomposer(
            [
                Identity2RZTranspiler(),
                H2RXRYTranspiler(),
                X2RXTranspiler(),
                Y2RYTranspiler(),
                Z2RZTranspiler(),
                SqrtX2RXTranspiler(),
                SqrtXdag2RXTranspiler(),
                SqrtY2RYTranspiler(),
                SqrtYdag2RYTranspiler(),
                S2RZTranspiler(),
                Sdag2RZTranspiler(),
                T2RZTranspiler(),
                Tdag2RZTranspiler(),
                U1ToRZTranspiler(),
                U2ToRXRZTranspiler(),
                U3ToRXRZTranspiler(),
                CZ2RXRYCNOTTranspiler(),
                SWAP2CNOTTranspiler(),
            ]
        ),
        FuseRotationTranspiler(),
    ]
)


class CliffordRZSetTranspiler(SequentialTranspiler):
    """CircuitTranspiler to transpile a QuntumCircuit into another
    QuantumCircuit containing only H, X, Y, Z, SqrtX, SqrtXdag, SqrtY,
    SqrtYdag, S, Sdg, RZ, CZ, and CNOT.

    Since this transpiler involves fusing rotation gates, converting
    rotation gates to named gates with a certain precision, and removing
    Identity gates, the action of the circuit before and after the
    conversion may not be completely equivalent.
    """

    def __init__(self, epsilon: float = 1.0e-9):
        super().__init__(
            [
                SingleQubitUnitaryMatrix2RYRZTranspiler(),
                TwoQubitUnitaryMatrixKAKTranspiler(),
                ParallelDecomposer(
                    [
                        PauliDecomposeTranspiler(),
                        PauliRotationDecomposeTranspiler(),
                        TOFFOLI2HTTdagCNOTTranspiler(),
                    ]
                ),
                ParallelDecomposer(
                    [
                        T2RZTranspiler(),
                        Tdag2RZTranspiler(),
                        RX2RZSqrtXTranspiler(),
                        RY2RZSqrtXTranspiler(),
                        U1ToRZTranspiler(),
                        U2ToRZSqrtXTranspiler(),
                        U3ToRZSqrtXTranspiler(),
                        SWAP2CNOTTranspiler(),
                    ]
                ),
                FuseRotationTranspiler(),
                RX2NamedTranspiler(epsilon),
                RY2NamedTranspiler(epsilon),
                RZ2NamedTranspiler(epsilon),
                IdentityEliminationTranspiler(),
            ]
        )


__all__ = [
    "CircuitTranspiler",
    "CircuitTranspilerProtocol",
    "GateDecomposer",
    "GateKindDecomposer",
    "ParallelDecomposer",
    "SequentialTranspiler",
    "RZSetTranspiler",
    "RotationSetTranspiler",
    "CliffordRZSetTranspiler",
    "CliffordApproximationTranspiler",
    "IdentityEliminationTranspiler",
    "IdentityInsertionTranspiler",
    "Identity2RZTranspiler",
    "PauliDecomposeTranspiler",
    "PauliRotationDecomposeTranspiler",
    "CNOT2CZHTranspiler",
    "CZ2CNOTHTranspiler",
    "CZ2RXRYCNOTTranspiler",
    "FuseRotationTranspiler",
    "NormalizeRotationTranspiler",
    "H2RXRYTranspiler",
    "H2RZSqrtXTranspiler",
    "QubitRemappingTranspiler",
    "RX2RZSqrtXTranspiler",
    "RY2RZSqrtXTranspiler",
    "RX2NamedTranspiler",
    "RY2NamedTranspiler",
    "RZ2NamedTranspiler",
    "S2RZTranspiler",
    "Sdag2RZTranspiler",
    "SingleQubitUnitaryMatrix2RYRZTranspiler",
    "SqrtX2RXTranspiler",
    "SqrtX2RZHTranspiler",
    "SqrtXdag2RXTranspiler",
    "SqrtXdag2RZSqrtXTranspiler",
    "SqrtY2RYTranspiler",
    "SqrtY2RZSqrtXTranspiler",
    "SqrtYdag2RYTranspiler",
    "SqrtYdag2RZSqrtXTranspiler",
    "SWAP2CNOTTranspiler",
    "T2RZTranspiler",
    "Tdag2RZTranspiler",
    "TOFFOLI2HTTdagCNOTTranspiler",
    "TwoQubitUnitaryMatrixKAKTranspiler",
    "U1ToRZTranspiler",
    "U2ToRXRZTranspiler",
    "U2ToRZSqrtXTranspiler",
    "U3ToRXRZTranspiler",
    "U3ToRZSqrtXTranspiler",
    "X2HZTranspiler",
    "X2RXTranspiler",
    "X2SqrtXTranspiler",
    "Y2RYTranspiler",
    "Y2RZXTranspiler",
    "Z2HXTranspiler",
    "Z2RZTranspiler",
    "su2_decompose",
    "su4_decompose",
]
