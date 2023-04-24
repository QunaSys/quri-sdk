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
    RX,
    RZ,
    H,
    Pauli,
    PauliRotation,
    QuantumCircuit,
    X,
    Y,
    Z,
)
from quri_parts.circuit.transpile import (
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
)


class TestMultiPauliDecompose:
    def test_pauli_decompose(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_gate(Pauli((1, 0, 2), (3, 1, 2)))
        transpiled = PauliDecomposeTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend([Z(1), X(0), Y(2)])

        assert transpiled.gates == expect.gates

    def test_pauli_rotation_decompose(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_gate(PauliRotation((0, 2, 1), (2, 3, 1), np.pi / 4.0))
        transpiled = PauliRotationDecomposeTranspiler()(circuit)

        expect = QuantumCircuit(3)
        gates = [
            RX(0, np.pi / 2.0),
            H(1),
            CNOT(1, 0),
            CNOT(2, 0),
            RZ(0, np.pi / 4.0),
            CNOT(2, 0),
            CNOT(1, 0),
            RX(0, -np.pi / 2.0),
            H(1),
        ]
        expect.extend(gates)

        assert transpiled.gates == expect.gates
