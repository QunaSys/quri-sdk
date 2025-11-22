# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np

from quri_parts.circuit import (
    CNOT,
    RX,
    RZ,
    H,
    ParametricQuantumCircuit,
    ParametricQuantumCircuitProtocol,
    ParametricQuantumGate,
    Pauli,
    PauliRotation,
    QuantumCircuit,
    QuantumGate,
    X,
    Y,
    Z,
)
from quri_parts.circuit.transpile import (
    ParametricPauliRotationDecomposeTranspiler,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
)


def _gates_close(
    x: Union[QuantumGate, ParametricQuantumGate],
    y: Union[QuantumGate, ParametricQuantumGate],
) -> bool:
    if isinstance(x, ParametricQuantumGate) and isinstance(y, ParametricQuantumGate):
        return x == y
    elif isinstance(x, QuantumGate) and isinstance(y, QuantumGate):
        return (
            x.name == y.name
            and x.target_indices == y.target_indices
            and x.control_indices == y.control_indices
            and np.allclose(x.params, y.params)
            and x.pauli_ids == y.pauli_ids
            and np.allclose(x.unitary_matrix, y.unitary_matrix)
        )
    else:
        return False


def _circuit_close(
    x: ParametricQuantumCircuitProtocol,
    y: ParametricQuantumCircuitProtocol,
) -> bool:
    return len(x.gates) == len(y.gates) and all(
        _gates_close(a, b) for a, b in zip(x.gates, y.gates)
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


class TestParametricMultiPauliDecompose:
    def test_parametric_pauli_rotation_decompose(self) -> None:
        circuit = ParametricQuantumCircuit(3)
        circuit.add_ParametricPauliRotation_gate((0, 2, 1), (2, 3, 1))
        transpiled = ParametricPauliRotationDecomposeTranspiler()(circuit)

        expect = ParametricQuantumCircuit(3)
        expect.extend(
            [
                RX(0, np.pi / 2.0),
                H(1),
                CNOT(1, 0),
                CNOT(2, 0),
            ]
        )
        expect.add_ParametricRZ_gate(0)
        expect.extend(
            [
                CNOT(2, 0),
                CNOT(1, 0),
                RX(0, -np.pi / 2.0),
                H(1),
            ]
        )

        assert _circuit_close(transpiled, expect)
