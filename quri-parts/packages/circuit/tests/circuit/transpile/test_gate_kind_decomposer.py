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
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    H,
    Identity,
    Pauli,
    PauliRotation,
    QuantumCircuit,
    QuantumGate,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    X,
    Y,
    Z,
    gate_names,
)
from quri_parts.circuit.transpile import (
    CliffordRZSetTranspiler,
    CZ2CNOTHTranspiler,
    CZ2RXRYCNOTTranspiler,
    H2RXRYTranspiler,
    H2RZSqrtXTranspiler,
    Identity2RZTranspiler,
    RotationSetTranspiler,
    RX2RZSqrtXTranspiler,
    RY2RZSqrtXTranspiler,
    RZSetTranspiler,
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
    STARSetTranspiler,
    SWAP2CNOTTranspiler,
    SWAPInsertionTranspiler,
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


def _gates_close(x: QuantumGate, y: QuantumGate) -> bool:
    return (
        x.name == y.name
        and x.target_indices == y.target_indices
        and x.control_indices == y.control_indices
        and np.allclose(x.params, y.params)
        and x.pauli_ids == y.pauli_ids
        and np.allclose(x.unitary_matrix, y.unitary_matrix)
    )


class TestFTQCSetTranspile:
    def test_cz_decompose(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_gate(CZ(0, 1))
        transpiled = CZ2CNOTHTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend([H(1), CNOT(0, 1), H(1)])

        assert transpiled.gates == expect.gates

    def test_swap_decompose(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_gate(SWAP(0, 1))
        transpiled = SWAP2CNOTTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend([CNOT(0, 1), CNOT(1, 0), CNOT(0, 1)])

        assert transpiled.gates == expect.gates

    def test_z2hx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(Z(0))
        transpiled = Z2HXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([H(0), X(0), H(0)])

        assert transpiled.gates == expect.gates

    def test_x2hz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(X(0))
        transpiled = X2HZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([H(0), Z(0), H(0)])

        assert transpiled.gates == expect.gates

    def test_x2sqrtx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(X(0))
        transpiled = X2SqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([SqrtX(0), SqrtX(0)])

        assert transpiled.gates == expect.gates

    def test_sqrtx2rzh_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtX(0))
        transpiled = SqrtX2RZHTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, -np.pi / 2.0), H(0), RZ(0, -np.pi / 2.0)])

        assert transpiled.gates == expect.gates


class TestRZSetTranspile:
    def test_identity2rz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(Identity(0))
        transpiled = Identity2RZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, 0.0)])

        assert transpiled.gates == expect.gates

    def test_h2rzsqrtx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(H(0))
        transpiled = H2RZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, np.pi / 2.0), SqrtX(0), RZ(0, np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_y2rzx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(Y(0))
        transpiled = Y2RZXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, -np.pi), X(0)])

        assert transpiled.gates == expect.gates

    def test_z2rz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(Z(0))
        transpiled = Z2RZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, np.pi)])

        assert transpiled.gates == expect.gates

    def test_sqrtxdag2rzsqrtx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtXdag(0))
        transpiled = SqrtXdag2RZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, -np.pi), SqrtX(0), RZ(0, -np.pi)])

        assert transpiled.gates == expect.gates

    def test_sqrty2rzsqrtx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtY(0))
        transpiled = SqrtY2RZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, -np.pi / 2.0), SqrtX(0), RZ(0, np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_sqrtydag2rzsqrtx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtYdag(0))
        transpiled = SqrtYdag2RZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, np.pi / 2.0), SqrtX(0), RZ(0, -np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_s2rz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(S(0))
        transpiled = S2RZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_sdag2rz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(Sdag(0))
        transpiled = Sdag2RZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, -np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_t2rz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(T(0))
        transpiled = T2RZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, np.pi / 4.0)])

        assert transpiled.gates == expect.gates

    def test_tdag2rz_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(Tdag(0))
        transpiled = Tdag2RZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, -np.pi / 4.0)])

        assert transpiled.gates == expect.gates

    def test_toffoli2httdagcnot_transpile(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_gate(TOFFOLI(0, 1, 2))
        transpiled = TOFFOLI2HTTdagCNOTTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend(
            [
                H(2),
                CNOT(1, 2),
                Tdag(2),
                CNOT(0, 2),
                T(2),
                CNOT(1, 2),
                Tdag(2),
                CNOT(0, 2),
                T(1),
                T(2),
                H(2),
                CNOT(0, 1),
                T(0),
                Tdag(1),
                CNOT(0, 1),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_rx2rzsqrtx_transpile(self) -> None:
        theta = np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(1)
        circuit.add_gate(RX(0, theta))
        transpiled = RX2RZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, np.pi / 2.0),
                SqrtX(0),
                RZ(0, theta + np.pi),
                SqrtX(0),
                RZ(0, np.pi / 2.0),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_ry2rzsqrtx_transpile(self) -> None:
        theta = np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(1)
        circuit.add_gate(RY(0, theta))
        transpiled = RY2RZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                SqrtX(0),
                RZ(0, theta + np.pi),
                SqrtX(0),
                RZ(0, np.pi),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_u1torz_transpile(self) -> None:
        lam = np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(1)
        circuit.add_gate(U1(0, lam))
        transpiled = U1ToRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, lam)])

        assert transpiled.gates == expect.gates

    def test_u2torzsqrtx_transpile(self) -> None:
        lam, phi = np.random.rand() * 2.0 * np.pi, np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(1)
        circuit.add_gate(U2(0, phi, lam))
        transpiled = U2ToRZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RZ(0, lam - np.pi / 2.0), SqrtX(0), RZ(0, phi + np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_u3torzsqrtx_transpiler(self) -> None:
        theta = np.random.rand() * 2.0 * np.pi
        phi = np.random.rand() * 2.0 * np.pi
        lam = np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(1)
        circuit.add_gate(U3(0, theta, phi, lam))
        transpiled = U3ToRZSqrtXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, lam),
                SqrtX(0),
                RZ(0, theta + np.pi),
                SqrtX(0),
                RZ(0, phi + np.pi),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_rzset_transpiler(self) -> None:
        theta = np.random.rand() * 2.0 * np.pi
        phi = np.random.rand() * 2.0 * np.pi
        lam = np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(3)
        circuit.extend(
            [
                Identity(2),
                X(0),
                Y(1),
                Z(2),
                H(0),
                SqrtXdag(1),
                SqrtY(2),
                SqrtYdag(0),
                S(1),
                Sdag(2),
                T(0),
                Tdag(1),
                RX(2, theta),
                RY(0, theta),
                U1(1, lam),
                U2(2, phi, lam),
                U3(0, theta, phi, lam),
                TOFFOLI(0, 1, 2),
                SWAP(0, 1),
                Pauli((0, 1, 2), (1, 2, 3)),
                PauliRotation((0, 1, 2), (1, 2, 3), theta),
            ]
        )
        transpiled = RZSetTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend(
            [
                RZ(2, 0.0),  # Idenitty
                X(0),  # X
                RZ(1, -np.pi),  # Y
                X(1),
                RZ(2, np.pi),  # Z
                RZ(0, np.pi / 2.0),  # H
                SqrtX(0),
                RZ(0, np.pi / 2.0),
                RZ(1, -np.pi),  # SqrtXdag
                SqrtX(1),
                RZ(1, -np.pi),
                RZ(2, -np.pi / 2.0),  # SqrtY
                SqrtX(2),
                RZ(2, np.pi / 2.0),
                RZ(0, np.pi / 2.0),  # SqrtYdag
                SqrtX(0),
                RZ(0, -np.pi / 2.0),
                RZ(1, np.pi / 2.0),  # S
                RZ(2, -np.pi / 2.0),  # Sdag
                RZ(0, np.pi / 4.0),  # T
                RZ(1, -np.pi / 4.0),  # Tdag
                RZ(2, np.pi / 2.0),  # RX
                SqrtX(2),
                RZ(2, theta + np.pi),
                SqrtX(2),
                RZ(2, np.pi / 2.0),
                SqrtX(0),  # RY
                RZ(0, theta + np.pi),
                SqrtX(0),
                RZ(0, np.pi),
                RZ(1, lam),  # U1
                RZ(2, lam - np.pi / 2.0),  # U2
                SqrtX(2),
                RZ(2, phi + np.pi / 2.0),
                RZ(0, lam),  # U3
                SqrtX(0),
                RZ(0, theta + np.pi),
                SqrtX(0),
                RZ(0, phi + np.pi),
                # TOFFOLI
                RZ(2, np.pi / 2.0),  # H
                SqrtX(2),
                RZ(2, np.pi / 2.0),
                CNOT(1, 2),
                RZ(2, -np.pi / 4.0),  # Tdag
                CNOT(0, 2),
                RZ(2, np.pi / 4.0),  # T
                CNOT(1, 2),
                RZ(2, -np.pi / 4.0),  # Tdag
                CNOT(0, 2),
                RZ(1, np.pi / 4.0),  # T
                RZ(2, np.pi * 3.0 / 4.0),  # T, H
                SqrtX(2),
                RZ(2, np.pi / 2.0),
                CNOT(0, 1),
                RZ(0, np.pi / 4.0),  # T
                RZ(1, -np.pi / 4.0),  # Tdag
                CNOT(0, 1),
                # Swap
                CNOT(0, 1),
                CNOT(1, 0),
                CNOT(0, 1),
                # Pauli
                X(0),  # X
                RZ(1, -np.pi),  # Y
                X(1),
                RZ(2, np.pi),  # Z
                # PauliRot
                RZ(0, np.pi / 2.0),  # H
                SqrtX(0),
                RZ(0, np.pi / 2.0),
                RZ(1, np.pi / 2.0),  # RX
                SqrtX(1),
                RZ(1, 3.0 * np.pi / 2.0),
                SqrtX(1),
                RZ(1, np.pi / 2.0),
                CNOT(2, 0),  # CNOT
                CNOT(1, 0),  # CNOT
                RZ(0, theta),  # RZ
                CNOT(1, 0),  # CNOT
                CNOT(2, 0),  # CNOT
                RZ(0, np.pi / 2.0),  # H
                SqrtX(0),
                RZ(0, np.pi / 2.0),
                RZ(1, np.pi / 2.0),  # RX
                SqrtX(1),
                RZ(1, np.pi / 2.0),
                SqrtX(1),
                RZ(1, np.pi / 2.0),
            ]
        )

        assert transpiled.gates == expect.gates


class TestRotationSetTranspile:
    def test_cz2rxrycnot_transpile(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_gate(CZ(0, 1))
        transpiled = CZ2RXRYCNOTTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                RY(1, np.pi / 2.0),
                RX(1, np.pi),
                CNOT(0, 1),
                RY(1, np.pi / 2.0),
                RX(1, np.pi),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_h2rxry_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(H(0))
        transpiled = H2RXRYTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RY(0, np.pi / 2.0), RX(0, np.pi)])

        assert transpiled.gates == expect.gates

    def test_x2rx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(X(0))
        transpiled = X2RXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RX(0, np.pi)])

        assert transpiled.gates == expect.gates

    def test_y2ry_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(Y(0))
        transpiled = Y2RYTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RY(0, np.pi)])

        assert transpiled.gates == expect.gates

    def test_sqrtx2rx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtX(0))
        transpiled = SqrtX2RXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RX(0, np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_sqrtxdag2rx_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtXdag(0))
        transpiled = SqrtXdag2RXTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RX(0, -np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_sqrty2ry_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtY(0))
        transpiled = SqrtY2RYTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RY(0, np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_sqrtydag2ry_transpile(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_gate(SqrtYdag(0))
        transpiled = SqrtYdag2RYTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend([RY(0, -np.pi / 2.0)])

        assert transpiled.gates == expect.gates

    def test_u2torxry_transpile(self) -> None:
        lam, phi = np.random.rand() * 2.0 * np.pi, np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(1)
        circuit.add_gate(U2(0, phi, lam))
        transpiled = U2ToRXRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [RZ(0, lam - np.pi / 2.0), RX(0, np.pi / 2.0), RZ(0, phi + np.pi / 2.0)]
        )

        assert transpiled.gates == expect.gates

    def test_u3torxrz_transpile(self) -> None:
        theta = np.random.rand() * 2.0 * np.pi
        phi = np.random.rand() * 2.0 * np.pi
        lam = np.random.rand() * 2.0 * np.pi

        circuit = QuantumCircuit(1)
        circuit.add_gate(U3(0, theta, phi, lam))
        transpiled = U3ToRXRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, lam),
                RX(0, np.pi / 2.0),
                RZ(0, theta + np.pi),
                RX(0, np.pi / 2.0),
                RZ(0, phi + np.pi),
            ]
        )

        assert transpiled.gates == expect.gates

    def test_rotationset_transpile(self) -> None:
        theta = np.pi / 7.0

        circuit = QuantumCircuit(3)
        circuit.add_Pauli_gate([2, 0, 1], [2, 3, 1])
        circuit.add_PauliRotation_gate([1, 0, 2], [1, 3, 2], theta)
        transpiled = RotationSetTranspiler()(circuit)

        target_set = {gate.name for gate in transpiled.gates}
        expect_set = {gate_names.RX, gate_names.RY, gate_names.RZ, gate_names.CNOT}

        assert target_set <= expect_set


class TestCliffordRZSetTranspile:
    def test_cliffordrzset_transpile(self) -> None:
        theta, phi, lam = np.pi / 3.0, np.pi / 5.0, np.pi / 7.0

        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                T(0),
                Tdag(0),
                RX(0, theta),
                RY(0, theta),
                U1(0, lam),
                U2(0, phi, lam),
                U3(0, theta, phi, lam),
            ]
        )
        transpiled = CliffordRZSetTranspiler()(circuit)

        expect = QuantumCircuit(3)
        expect.extend(
            [
                S(0),  # T, Tdag, RX
                SqrtX(0),
                RZ(0, theta + np.pi),
                SqrtX(0),
                S(0),
                SqrtX(0),  # RY
                RZ(0, theta + np.pi),
                SqrtX(0),
                RZ(0, np.pi / 2.0 + 2.0 * lam),  # U1, U2
                SqrtX(0),
                RZ(0, phi + lam + np.pi / 2.0),  # U2, U3
                SqrtX(0),
                RZ(0, theta + np.pi),
                SqrtX(0),
                RZ(0, phi + np.pi),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            assert _gates_close(t, e)


class TestSWAPInsertionTranspiler:
    def test_ordered_connectivity_transpile(self) -> None:
        qubit_count = 7
        transpiler = SWAPInsertionTranspiler()
        expected_gates = (
            (CNOT(0, 1),),
            (
                SWAP(0, 1),
                CNOT(1, 2),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                CNOT(2, 3),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                SWAP(2, 3),
                CNOT(3, 4),
                SWAP(2, 3),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                SWAP(2, 3),
                SWAP(3, 4),
                CNOT(4, 5),
                SWAP(3, 4),
                SWAP(2, 3),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                SWAP(2, 3),
                SWAP(3, 4),
                SWAP(4, 5),
                CNOT(5, 6),
                SWAP(4, 5),
                SWAP(3, 4),
                SWAP(2, 3),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
        )
        for j in range(1, qubit_count):
            circuit = QuantumCircuit(qubit_count)
            circuit.add_CNOT_gate(0, j)

            target_circuit = QuantumCircuit(qubit_count, gates=expected_gates[j - 1])

            transpiled = transpiler(circuit)
            assert transpiled.gates == target_circuit.gates

    def test_reverse_ordered_connectivity_transpile(self) -> None:
        qubit_count = 7
        transpiler = SWAPInsertionTranspiler()
        expected_gates = (
            (CNOT(1, 0),),
            (
                SWAP(0, 1),
                CNOT(2, 1),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                CNOT(3, 2),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                SWAP(2, 3),
                CNOT(4, 3),
                SWAP(2, 3),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                SWAP(2, 3),
                SWAP(3, 4),
                CNOT(5, 4),
                SWAP(3, 4),
                SWAP(2, 3),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
            (
                SWAP(0, 1),
                SWAP(1, 2),
                SWAP(2, 3),
                SWAP(3, 4),
                SWAP(4, 5),
                CNOT(6, 5),
                SWAP(4, 5),
                SWAP(3, 4),
                SWAP(2, 3),
                SWAP(1, 2),
                SWAP(0, 1),
            ),
        )
        for j in range(1, qubit_count):
            circuit = QuantumCircuit(qubit_count)
            circuit.add_CNOT_gate(j, 0)

            target_circuit = QuantumCircuit(qubit_count, gates=expected_gates[j - 1])

            transpiled = transpiler(circuit)
            assert transpiled.gates == target_circuit.gates


class TestSTARSetTranspile:
    def test_starset_transpile(self) -> None:
        theta = np.pi / 7.0

        circuit = QuantumCircuit(3)
        circuit.add_Pauli_gate([2, 0, 1], [2, 3, 1])
        circuit.add_PauliRotation_gate([1, 0, 2], [1, 3, 2], theta)
        transpiled = STARSetTranspiler()(circuit)

        target_set = {gate.name for gate in transpiled.gates}
        expect_set = {
            gate_names.H,
            gate_names.S,
            gate_names.RZ,
            gate_names.CNOT,
        }

        assert target_set <= expect_set
