# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.circuit import (
    CNOT,
    CZ,
    RX,
    RY,
    U2,
    U3,
    H,
    Identity,
    PauliRotation,
    QuantumCircuit,
    S,
    SqrtX,
    SqrtXdag,
    SqrtY,
    T,
    Z,
)
from quri_parts.circuit.transpile import CliffordApproximationTranspiler


class TestCliffordApproximationTranspile:
    def test_cliffordapproximation_transpiler(self) -> None:
        gate_list = [
            H(0),
            T(1),
            RY(0, 1 * np.pi / 2),
            RX(1, 1.4 * np.pi / 2),
            U2(1, 0.5 * np.pi / 2, 1.7 * np.pi / 2),
            U3(3, 0.5 * np.pi / 2, 1.5 * np.pi / 2, 2.5 * np.pi / 2),
            CZ(2, 3),
            PauliRotation((0, 1, 2), (3, 2, 1), 0.51 * np.pi / 2),
        ]
        circuit = QuantumCircuit(4, gates=gate_list)
        transpiled = CliffordApproximationTranspiler()(circuit)

        expect = QuantumCircuit(4)
        expect.extend(
            [
                H(0),  # H(0)
                S(1),  # T(1)
                SqrtY(0),  # RY
                SqrtX(1),  # RX
                S(1),  # U2
                SqrtX(1),
                Z(1),
                Z(3),  # U3
                SqrtX(3),
                Z(3),
                SqrtX(3),
                Identity(3),
                CZ(2, 3),  # CZ
                SqrtX(1),  # PauliRotation
                H(2),
                CNOT(2, 0),
                CNOT(1, 0),
                S(0),
                CNOT(1, 0),
                CNOT(2, 0),
                SqrtXdag(1),
                H(2),
            ]
        )

        assert transpiled.gates == expect.gates
