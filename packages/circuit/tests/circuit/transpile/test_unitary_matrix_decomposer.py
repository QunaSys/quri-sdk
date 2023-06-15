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

from quri_parts.circuit import CNOT, RX, RY, RZ, H, QuantumCircuit, QuantumGate, S
from quri_parts.circuit.transpile import (
    SingleQubitUnitaryMatrix2RYRZTranspiler,
    TwoQubitUnitaryMatrixKAKTranspiler,
)


def _assert_params_close(x: QuantumGate, y: QuantumGate) -> None:
    assert (
        x.name == y.name
        and x.target_indices == y.target_indices
        and x.control_indices == y.control_indices
        and np.allclose(x.params, y.params)
        and x.pauli_ids == y.pauli_ids
        and x.unitary_matrix == y.unitary_matrix
    )


class TestSingleQubitDecompose:
    def test_id_decompose(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_SingleQubitUnitaryMatrix_gate(0, [[1, 0], [0, 1]])
        transpiled = SingleQubitUnitaryMatrix2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, 0.0),
                RY(0, 0.0),
                RZ(0, 0.0),
            ]
        )

        assert transpiled == expect

    def test_x_decompose(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_SingleQubitUnitaryMatrix_gate(0, [[0, 1], [1, 0]])
        transpiled = SingleQubitUnitaryMatrix2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, np.pi),
                RY(0, np.pi),
                RZ(0, 0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)

    def test_y_decompose(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_SingleQubitUnitaryMatrix_gate(0, [[0, -1j], [1j, 0]])
        transpiled = SingleQubitUnitaryMatrix2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, 0),
                RY(0, np.pi),
                RZ(0, 0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)

    def test_h_decompose(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_SingleQubitUnitaryMatrix_gate(
            0, 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        )
        transpiled = SingleQubitUnitaryMatrix2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, np.pi),
                RY(0, np.pi / 2),
                RZ(0, 0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)

    def test_t_decompose(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_SingleQubitUnitaryMatrix_gate(
            0, [[1, 0], [0, np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4)]]
        )
        transpiled = SingleQubitUnitaryMatrix2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, np.pi / 4),
                RY(0, 0),
                RZ(0, 0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)

    def test_fixed_decompose(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.add_SingleQubitUnitaryMatrix_gate(
            0,
            [
                [-0.74161209 - 0.58511358j, 0.32678536 - 0.02940967j],
                [-0.14825031 - 0.29270369j, -0.88922216 - 0.31879516j],
            ],
        )
        transpiled = SingleQubitUnitaryMatrix2RYRZTranspiler()(circuit)

        expect = QuantumCircuit(1)
        expect.extend(
            [
                RZ(0, 5.525447921071612),
                RY(0, 0.6685959371692511),
                RZ(0, 0.43399113617430446),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)


class TestTwoQubitDecompose:
    def test_swap_decompose(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_TwoQubitUnitaryMatrix_gate(
            1,
            0,
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
        )
        transpiled = TwoQubitUnitaryMatrixKAKTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                RZ(0, -np.pi),
                RY(0, np.pi),
                RZ(0, 0),
                RZ(1, -np.pi),
                RY(1, np.pi),
                RZ(1, 0),
                CNOT(0, 1),
                RX(0, np.pi / 2),
                H(0),
                RZ(1, -np.pi / 2),
                CNOT(0, 1),
                S(0),
                H(0),
                RZ(1, np.pi / 2),
                CNOT(0, 1),
                RX(0, -np.pi / 2),
                RZ(0, 0),
                RY(0, 0),
                RZ(0, 0),
                RX(1, np.pi / 2),
                RZ(1, 0),
                RY(1, 0),
                RZ(1, 0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)

    def test_cz_decompose(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_TwoQubitUnitaryMatrix_gate(
            1,
            0,
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
        )
        transpiled = TwoQubitUnitaryMatrixKAKTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                RZ(0, -np.pi),
                RY(0, np.pi),
                RZ(0, 0),
                RZ(1, -np.pi),
                RY(1, np.pi),
                RZ(1, 0),
                CNOT(0, 1),
                RX(0, 0),
                H(0),
                RZ(1, -np.pi / 2),
                CNOT(0, 1),
                S(0),
                H(0),
                RZ(1, 0),
                CNOT(0, 1),
                RX(0, -np.pi / 2),
                RZ(0, -np.pi * 3 / 2),
                RY(0, np.pi),
                RZ(0, 0),
                RX(1, np.pi / 2),
                RZ(1, np.pi / 2),
                RY(1, np.pi),
                RZ(1, 0),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)

    def test_fixed_decompose(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_TwoQubitUnitaryMatrix_gate(
            1,
            0,
            [
                [
                    -0.52037119 - 0.23369154j,
                    0.00336457 - 0.67280194j,
                    0.21766929 - 0.19839962j,
                    0.06844591 - 0.36124943j,
                ],
                [
                    -0.1202352 - 0.34198246j,
                    -0.51915342 + 0.13693515j,
                    0.17517988 - 0.24753382j,
                    0.5140112 + 0.4734464j,
                ],
                [
                    -0.14712195 - 0.51048619j,
                    -0.36435886 + 0.28986008j,
                    0.07735599 + 0.56042397j,
                    -0.28453794 - 0.31616754j,
                ],
                [
                    -0.34491589 - 0.37680976j,
                    0.20500064 + 0.01591263j,
                    -0.68754106 - 0.16889347j,
                    -0.28612016 + 0.33714463j,
                ],
            ],
        )
        transpiled = TwoQubitUnitaryMatrixKAKTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                RZ(0, 1.964151658149481),
                RY(0, 2.445734665400974),
                RZ(0, -1.2228580059647152),
                RZ(1, -2.0261768746801208),
                RY(1, 2.23384008194361),
                RZ(1, 0.542716624931449),
                CNOT(0, 1),
                RX(0, -1.3433479167618008),
                H(0),
                RZ(1, -0.1535936645886139),
                CNOT(0, 1),
                S(0),
                H(0),
                RZ(1, -0.7600483917018804),
                CNOT(0, 1),
                RX(0, -np.pi / 2),
                RZ(0, 4.543181270456171),
                RY(0, 2.4523790983331857),
                RZ(0, 0.18060536576440683),
                RX(1, np.pi / 2),
                RZ(1, 1.6106757636768554),
                RY(1, 0.8382960839329676),
                RZ(1, -1.0044839579124587),
            ]
        )

        for t, e in zip(transpiled.gates, expect.gates):
            _assert_params_close(t, e)
