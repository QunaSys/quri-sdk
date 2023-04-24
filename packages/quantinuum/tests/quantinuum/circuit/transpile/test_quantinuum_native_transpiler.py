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

from quri_parts.circuit import CNOT, RZ, H, QuantumCircuit
from quri_parts.quantinuum.circuit import RZZ, ZZ, U1q
from quri_parts.quantinuum.circuit.transpile import (
    CNOT2U1qZZRZTranspiler,
    CNOTRZ2RZZTranspiler,
    CZ2RZZZTranspiler,
    H2U1qRZTranspiler,
    RX2U1qTranspiler,
    RY2U1qTranspiler,
    U1qNormalizeWithRZTranspiler,
)


class TestQuantinuumNativeTranspile:
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

    def test_u1qnormalize_transpile(self) -> None:
        phi = np.random.rand()
        theta = np.pi / 5.0

        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                U1q(0, 0.0, phi),
                U1q(0, -np.pi / 2.0, phi),
                U1q(0, np.pi, phi),
                U1q(0, np.pi / 2.0, phi),
                U1q(0, theta, phi),
            ]
        )
        transpiled = U1qNormalizeWithRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, 0.0),
                U1q(0, np.pi / 2.0, phi + np.pi),
                U1q(0, np.pi, phi),
                U1q(0, np.pi / 2.0, phi),
                U1q(0, np.pi / 2.0, phi + np.pi / 2.0),
                RZ(0, theta),
                U1q(0, np.pi / 2.0, phi - np.pi / 2.0),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_cz2rzz_transpile(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_CZ_gate(0, 1)
        transpiled = CZ2RZZZTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                RZ(0, -np.pi / 2.0),
                RZ(1, -np.pi / 2.0),
                ZZ(0, 1),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_cnotrz2rzz_transpile(self) -> None:
        theta0, theta1, theta2 = np.random.rand(3)

        circuit = QuantumCircuit(3)
        circuit.extend(
            [
                CNOT(0, 1),
                RZ(1, theta0),
                CNOT(0, 1),
                H(2),
                CNOT(1, 2),
                RZ(1, theta1),
                CNOT(1, 2),
                CNOT(2, 0),
                RZ(0, theta2),
                CNOT(2, 0),
                H(1),
            ]
        )
        transpiled = CNOTRZ2RZZTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend(
            [
                RZZ(0, 1, theta0),
                H(2),
                CNOT(1, 2),
                RZ(1, theta1),
                CNOT(1, 2),
                RZZ(2, 0, theta2),
                H(1),
            ]
        )

        assert transpiled.gates == expect.gates
