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
import pytest

from quri_parts.circuit import (
    CNOT,
    RX,
    RY,
    U3,
    H,
    Pauli,
    PauliRotation,
    QuantumCircuit,
    X,
)
from quri_parts.circuit.transpile import QubitRemappingTranspiler


class TestQubitRemapping:
    def test_qubit_remapping(self) -> None:
        circuit = QuantumCircuit(5)
        circuit.extend(
            [
                H(0),
                CNOT(2, 3),
                X(1),
                H(3),
                RX(2, np.pi / 4),
                RY(1, np.pi / 2),
                U3(0, np.pi / 2, np.pi / 4, np.pi / 8),
                Pauli((1, 3), (1, 3)),
                PauliRotation((2, 0), (3, 2), np.pi / 4),
            ]
        )

        qubit_mapping = {0: 3, 1: 2, 2: 7, 3: 6, 4: 8}
        transpiled = QubitRemappingTranspiler(qubit_mapping)(circuit)

        assert transpiled.qubit_count == 9

        expected_gates = (
            H(3),
            CNOT(7, 6),
            X(2),
            H(6),
            RX(7, np.pi / 4),
            RY(2, np.pi / 2),
            U3(3, np.pi / 2, np.pi / 4, np.pi / 8),
            Pauli((2, 6), (1, 3)),
            PauliRotation((7, 3), (3, 2), np.pi / 4),
        )

        assert transpiled.gates == expected_gates

    def test_error_on_duplicate_indices(self) -> None:
        with pytest.raises(ValueError):
            QubitRemappingTranspiler({0: 1, 1: 2, 2: 2})

    def test_error_on_unspecified_index(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.extend([H(0), X(2)])

        qubit_mapping = {0: 1, 1: 2}
        transpiler = QubitRemappingTranspiler(qubit_mapping)

        with pytest.raises(ValueError):
            transpiler(circuit)
