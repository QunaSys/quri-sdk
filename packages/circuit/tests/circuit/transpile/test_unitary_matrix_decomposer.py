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

from quri_parts.circuit import RY, RZ, QuantumCircuit, QuantumGate
from quri_parts.circuit.transpile import SingleQubitUnitaryMatrix2RYRZTranspiler


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
