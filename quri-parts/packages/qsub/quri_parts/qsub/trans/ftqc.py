# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import gate_names
from quri_parts.circuit.transpile import (
    CliffordConversionTranspiler,
    CZ2CNOTHTranspiler,
    FuseRotationTranspiler,
    IdentityEliminationTranspiler,
    ParallelDecomposer,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    RX2RZHTranspiler,
    RY2RZHTranspiler,
    RZ2NamedTranspiler,
    SequentialTranspiler,
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    SWAP2CNOTTranspiler,
    Tdag2STTranspiler,
    TOFFOLI2HTTdagCNOTTranspiler,
    TwoQubitUnitaryMatrixKAKTranspiler,
    U1ToRZTranspiler,
)
from quri_parts.circuit.transpile.rz2hst import RZ2HSTTranspiler
from quri_parts.qsub.sub import Sub

from .qp_trans import SeparateQURIPartsTranspiler
from .transpiler import SubTranspilerProtocol


class FTQCSetTranspiler(SubTranspilerProtocol):
    """A qsub Sub transpiler that rewrites a ``Sub`` into the FTQC basic gate
    set {H, S, T, CNOT}.

    RZ angles that are multiples of pi/4 map exactly onto {H, S, T};
    other angles are approximated with gridsynth (via ``pygridsynth``)
    to a precision of ``epsilon`` (default ``1e-9``).
    """

    def __init__(self, epsilon: float = 1.0e-9) -> None:
        self._transpiler = SeparateQURIPartsTranspiler(
            [
                SequentialTranspiler(
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
                        CZ2CNOTHTranspiler(),
                        SWAP2CNOTTranspiler(),
                        ParallelDecomposer(
                            [
                                RX2RZHTranspiler(),
                                RY2RZHTranspiler(),
                                U1ToRZTranspiler(),
                            ]
                        ),
                        CliffordConversionTranspiler((gate_names.H, gate_names.S)),
                        FuseRotationTranspiler(),
                        RZ2NamedTranspiler(epsilon, allow_t_tdag=True),
                        RZ2HSTTranspiler(epsilon),
                        FuseRotationTranspiler(),
                        CliffordConversionTranspiler((gate_names.H, gate_names.S)),
                        Tdag2STTranspiler(),
                        IdentityEliminationTranspiler(),
                    ]
                )
            ]
        )

    def __call__(self, sub: Sub) -> Sub:
        return self._transpiler(sub)
