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

from quri_parts.circuit import RZ, QuantumCircuit
from quri_parts.honeywell.circuit import ZZ, U1q
from quri_parts.honeywell.circuit.transpile import (
    CNOT2U1qZZRZTranspiler,
    H2U1qRZTranspiler,
    RX2U1qTranspiler,
    RY2U1qTranspiler,
)


class TestHoneywellNativeTranspile:
    def test_rx2u1q_transpile(self) -> None:
        theta = np.random.rand()
        circuit = QuantumCircuit(1)
        circuit.add_RX_gate(0, theta)
        transpiled = RX2U1qTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([U1q(0, theta, 0.0)])

        assert transpiled.gates == expect.gates

    def test_ry2u1q_transpile(self) -> None:
        theta = np.random.rand()
        circuit = QuantumCircuit(1)
        circuit.add_RY_gate(0, theta)
        transpiled = RY2U1qTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([U1q(0, theta, np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_h2u1qrz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_H_gate(0)
        transpiled = H2U1qRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([U1q(0, np.pi / 2.0, -np.pi / 2.0), RZ(0, np.pi)])

        assert transpiled.gates == expect.gates

    def test_cnot2u1qzzrz_transpile(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_CNOT_gate(0, 1)
        transpiled = CNOT2U1qZZRZTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                U1q(1, -np.pi / 2.0, np.pi / 2.0),
                ZZ(0, 1),
                RZ(0, -np.pi / 2.0),
                U1q(1, np.pi / 2.0, np.pi),
                RZ(1, -np.pi / 2.0),
            ]
        )

        assert transpiled.gates == expect.gates
