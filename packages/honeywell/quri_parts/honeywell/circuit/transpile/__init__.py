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

from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    ParallelDecomposer,
    RotationSetTranspiler,
    SequentialTranspiler,
)

from .honeywell_native_transpiler import (
    CNOT2U1qZZRZTranspiler,
    H2U1qRZTranspiler,
    RX2U1qTranspiler,
    RY2U1qTranspiler,
)

#: CircuitTranspiler to transpile a QuantumCircuit into another
#: QuantumCircuit containing only U1q, RZ, ZZ, and RZZ gates.
#: Note that the converted circuit contains Honeywell native gates.
HoneywellSetTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [
        RotationSetTranspiler(),
        ParallelDecomposer(
            [
                RX2U1qTranspiler(),
                RY2U1qTranspiler(),
                CNOT2U1qZZRZTranspiler(),
            ]
        ),
    ]
)


__all__ = [
    "HoneywellSetTranspiler",
    "CNOT2U1qZZRZTranspiler",
    "H2U1qRZTranspiler",
    "RX2U1qTranspiler",
    "RY2U1qTranspiler",
]
