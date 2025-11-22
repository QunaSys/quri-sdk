# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast

import numpy as np
import numpy.typing as npt
import qulacs

from quri_parts.circuit import (
    CNOT,
    RX,
    RY,
    RZ,
    H,
    ImmutableQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    S,
    T,
    X,
    gate_names,
)
from quri_parts.ionq.circuit import MS, XX, GPi, GPi2
from quri_parts.ionq.circuit import gate_names as ionq_gate_names
from quri_parts.ionq.circuit.transpile import (
    CNOT2RXRYXXTranspiler,
    IonQNativeTranspiler,
    IonQSetTranspiler,
)
from quri_parts.qulacs.circuit import convert_circuit


def ionq_gate_matrix(gate: QuantumGate) -> npt.NDArray[np.complex128]:
    if gate.name == ionq_gate_names.GPi:
        phi = gate.params[0] * 2 * np.pi
        ret = np.array(
            [
                [0, np.cos(phi) - 1j * np.sin(phi)],
                [np.cos(phi) + 1j * np.sin(phi), 0],
            ]
        )
    elif gate.name == ionq_gate_names.GPi2:
        phi = gate.params[0] * 2 * np.pi
        ret = (
            1
            / np.sqrt(2)
            * np.array(
                [
                    [1, -1j * (np.cos(phi) - 1j * np.sin(phi))],
                    [-1j * (np.cos(phi) + 1j * np.sin(phi)), 1],
                ]
            )
        )
    elif gate.name == ionq_gate_names.XX:
        phi = gate.params[0]
        ret = np.array(
            [
                [np.cos(phi), 0, 0, -1j * np.sin(phi)],
                [0, np.cos(phi), -1j * np.sin(phi), 0],
                [0, -1j * np.sin(phi), np.cos(phi), 0],
                [-1j * np.sin(phi), 0, 0, np.cos(phi)],
            ]
        )
    elif gate.name == ionq_gate_names.MS:
        phi0, phi1 = gate.params[0] * 2 * np.pi, gate.params[1] * 2 * np.pi
        ret = (
            1
            / np.sqrt(2)
            * np.array(
                [
                    [1, 0, 0, -1j * (np.cos(phi0 + phi1) - 1j * np.sin(phi0 + phi1))],
                    [0, 1, -1j * (np.cos(phi0 - phi1) - 1j * np.sin(phi0 - phi1)), 0],
                    [0, -1j * (np.cos(phi0 - phi1) + 1j * np.sin(phi0 - phi1)), 1, 0],
                    [-1j * (np.cos(phi0 + phi1) + 1j * np.sin(phi0 + phi1)), 0, 0, 1],
                ]
            )
        )
    else:
        raise NotImplementedError(f"{gate.name} is not supported.")

    return cast(npt.NDArray[np.complex128], ret)


def ionq_circuit_state(
    circuit: ImmutableQuantumCircuit,
) -> npt.NDArray[np.complex128]:
    ionq_gates_1 = [ionq_gate_names.GPi, ionq_gate_names.GPi2]
    ionq_gates_2 = [ionq_gate_names.XX, ionq_gate_names.MS]
    qs_circuit = qulacs.QuantumCircuit(circuit.qubit_count)
    for gate in circuit.gates:
        if gate.name in ionq_gates_1:
            # We need to disable type check due to an error in qulacs type annotation
            # https://github.com/qulacs/qulacs/issues/537
            qs_circuit.add_dense_matrix_gate(
                gate.target_indices[0], ionq_gate_matrix(gate)
            )
        elif gate.name in ionq_gates_2:
            # We need to disable type check due to an error in qulacs type annotation
            # https://github.com/qulacs/qulacs/issues/537
            qs_circuit.add_dense_matrix_gate(
                [gate.target_indices[0], gate.target_indices[1]],
                ionq_gate_matrix(gate),  # noqa: E501
            )
        elif gate.name == gate_names.RX:
            qs_circuit.add_RX_gate(gate.target_indices[0], -gate.params[0])
        elif gate.name == gate_names.RY:
            qs_circuit.add_RY_gate(gate.target_indices[0], -gate.params[0])
        elif gate.name == gate_names.RZ:
            qs_circuit.add_RZ_gate(gate.target_indices[0], -gate.params[0])
        else:
            raise NotImplementedError(f"{gate.name} is not supported.")
    state = qulacs.QuantumState(circuit.qubit_count)
    qs_circuit.update_quantum_state(state)
    return cast(npt.NDArray[np.complex128], state.get_vector())


def qulacs_circuit_state(circuit: QuantumCircuit) -> npt.NDArray[np.complex128]:
    qs_circuit = convert_circuit(circuit)
    state = qulacs.QuantumState(circuit.qubit_count)
    qs_circuit.update_quantum_state(state)
    return cast(npt.NDArray[np.complex128], state.get_vector())


def assert_state_elem_equal(
    xs: npt.NDArray[np.complex128], ys: npt.NDArray[np.complex128]
) -> None:
    """Only the norm of each element is compared between given two state
    vectors.

    Even if pass the test, the value of each element may not be
    identical, but the measurement results will be identical.
    """
    assert np.allclose(np.absolute(xs), np.absolute(ys))


class TestIonQNativeTranspile:
    def test_ionqset_transpile(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.extend([H(0), CNOT(0, 1), S(1), H(1), CNOT(2, 0), H(2), T(1), X(0)])
        transpiled = IonQSetTranspiler()(circuit)

        state = qulacs_circuit_state(circuit)
        transpiled_state = ionq_circuit_state(transpiled)
        assert_state_elem_equal(state, transpiled_state)

    def test_ionqnative_transpile_phase(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.extend(
            [
                RY(0, np.pi / 5.0),
                RX(0, -np.pi / 2.0),
                RZ(0, -np.pi / 7.0),
                RY(0, np.pi / 2.0),
                RY(0, -np.pi),
                RX(0, np.pi / 2.0),
                RY(0, np.pi),
                RZ(0, np.pi / 4.0),
                RX(0, np.pi / 11.0),
                RY(0, -np.pi / 2.0),
                RX(0, np.pi),
                RY(0, np.pi / 3.0),
                RX(0, -np.pi),
                RZ(0, np.pi),
                RX(0, np.pi / 2.0),
            ]
        )
        transpiled = IonQNativeTranspiler()(circuit)

        state = ionq_circuit_state(circuit)
        transpiled_state = ionq_circuit_state(transpiled)
        assert_state_elem_equal(state, transpiled_state)

    def test_ionqnative_transpile_no_phase(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.extend(
            [
                RX(0, np.pi / 2.0),  # RX
                RX(0, np.pi),
                RX(0, -np.pi / 2.0),
                RX(0, -np.pi),
                RY(0, np.pi / 2.0),  # RY
                RY(0, np.pi),
                RY(0, -np.pi / 2.0),
                RY(0, -np.pi),
                RY(0, np.pi / 2.0),  # CNOT
                XX(0, 1, np.pi / 4.0),
                RX(0, -np.pi / 2.0),
                RX(1, -np.pi / 2.0),
                RY(0, -np.pi / 2.0),
            ]
        )
        transpiled = IonQNativeTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                GPi2(0, 0.0),  # RX
                GPi(0, 0.0),
                GPi2(0, 0.5),
                GPi(0, 0.5),
                GPi2(0, 0.25),  # RY
                GPi(0, 0.25),
                GPi2(0, 0.75),
                GPi(0, 0.75),
                GPi2(0, 0.25),  # CNOT
                MS(0, 1, 0.0, 0.0),
                GPi2(0, 0.5),
                GPi2(1, 0.5),
                GPi2(0, 0.75),
            ]
        )

        assert transpiled.gates == expect.gates

        state = ionq_circuit_state(circuit)
        transpiled_state = ionq_circuit_state(transpiled)
        assert_state_elem_equal(state, transpiled_state)

    def test_cnot2rxryxx_transpile(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_gate(CNOT(0, 1))
        transpiled = CNOT2RXRYXXTranspiler()(circuit)

        expect = QuantumCircuit(2)
        expect.extend(
            [
                RY(0, np.pi / 2.0),
                XX(0, 1, np.pi / 4.0),
                RX(0, -np.pi / 2.0),
                RX(1, -np.pi / 2.0),
                RY(0, -np.pi / 2.0),
            ]
        )

        assert transpiled.gates == expect.gates

        state = qulacs_circuit_state(circuit)
        transpiled_state = ionq_circuit_state(transpiled)
        assert_state_elem_equal(state, transpiled_state)
