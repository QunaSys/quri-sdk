# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Iterable, cast

import numpy as np
import pytest
from typing_extensions import TypeAlias

from quri_parts.circuit import H, ImmutableQuantumCircuit, Pauli, QuantumGate, X, Y, Z
from quri_parts.core.state import (
    ComputationalBasisState,
    StateVectorType,
    comp_basis_superposition,
)
from quri_parts.core.state.state import GeneralCircuitQuantumState

from .helpers import raw_vector


def a_state() -> ComputationalBasisState:
    s = ComputationalBasisState(2)
    s = s.with_pauli_gate_applied(X(0))
    s = s.with_pauli_gate_applied(Z(1))
    return s


class TestComputationalBasisState:
    def test_circuit(self) -> None:
        state = a_state()
        circuit = state.circuit
        assert isinstance(circuit, ImmutableQuantumCircuit)
        assert len(circuit.gates) == 1

        circuit_copy = circuit.get_mutable_copy()
        circuit_copy.add_X_gate(0)
        assert state.circuit != circuit_copy
        assert len(state.circuit.gates) == 1

    def test_qubit_count(self) -> None:
        state = a_state()
        assert state.qubit_count == 2

    def test_bits(self) -> None:
        state = ComputationalBasisState(3, bits=0b110)
        assert state.bits == 6
        state = state.with_pauli_gate_applied(X(0))
        assert state.bits == 7
        state = state.with_pauli_gate_applied(Z(0))
        assert state.bits == 7
        state = state.with_pauli_gate_applied(Y(0))
        assert state.bits == 6
        state = state.with_pauli_gate_applied(X(0))
        assert state.bits == 7

    def test_phase(self) -> None:
        state = ComputationalBasisState(1)
        state = state.with_pauli_gate_applied(X(0))
        assert state.phase == pytest.approx(0)
        state = state.with_pauli_gate_applied(Z(0))
        assert state.phase == pytest.approx(np.pi)
        state = state.with_pauli_gate_applied(Y(0))
        assert state.phase == pytest.approx(1 / 2 * np.pi)
        state = state.with_pauli_gate_applied(X(0))
        assert state.phase == pytest.approx(1 / 2 * np.pi)

    def test_with_pauli_gate_applied(self) -> None:
        state = ComputationalBasisState(2)
        # X0 Z0 Y1 Y0 Y1 = -1j (global phase will be ignored)
        gates: Iterable[QuantumGate] = (X(0), Z(0), Y(1), Pauli((0, 1), (2, 2)))
        for g in gates:
            state = state.with_pauli_gate_applied(g)
        circuit = state.circuit
        assert circuit.gates == ()

    def test_apply_with_out_of_index(self) -> None:
        state = ComputationalBasisState(2)
        with pytest.raises(ValueError):
            state = state.with_pauli_gate_applied(X(2))

    def test_with_gates_applied_pauli(self) -> None:
        state = ComputationalBasisState(2, bits=0b10)
        state2 = state.with_gates_applied([Y(0), X(1)])

        assert state != state2
        assert state.bits == 0b10
        assert state.phase == 0
        assert isinstance(state2, ComputationalBasisState)
        assert state2.bits == 0b01
        assert state2.phase == pytest.approx(1 / 2 * np.pi)

    def test_with_gates_applied_general(self) -> None:
        state = ComputationalBasisState(2, bits=0b10)
        state2 = state.with_gates_applied([H(0), X(1)])

        assert state != state2
        assert state.bits == 0b10
        assert state.phase == 0
        assert isinstance(state2, GeneralCircuitQuantumState)
        circuit = state.circuit.get_mutable_copy()
        circuit.add_H_gate(0)
        circuit.add_X_gate(1)
        assert state2.circuit == circuit


Matrix: TypeAlias = "np.typing.NDArray[np.complex128]"

_I: Matrix = np.array([[1, 0], [0, 1]])
_X: Matrix = np.array([[0, 1], [1, 0]])
_Z: Matrix = np.array([[1, 0], [0, -1]])
_Y: Matrix = np.array([[0, -1j], [1j, 0]])
GATES_AND_MATRICES = cast(
    list[tuple[QuantumGate, Matrix]],
    [
        (X(1), np.kron(_X, _I)),
        (Y(1), np.kron(_Y, _I)),
        (Z(1), np.kron(_Z, _I)),
        (X(0), np.kron(_I, _X)),
        (Y(0), np.kron(_I, _Y)),
        (Z(0), np.kron(_I, _Z)),
    ],
)


def get_random_gates_and_vector() -> tuple[list[QuantumGate], StateVectorType]:
    """Returns Pauli gates chosen randomly and a state created by applying the
    gates to |0> state."""
    chosen_gates = random.choices(GATES_AND_MATRICES, k=random.randint(0, 5))
    circuit = []
    vector: StateVectorType = np.array([1, 0, 0, 0])
    for gate, matrix in chosen_gates:
        circuit.append(gate)
        vector = matrix @ vector
    return circuit, vector


class TestSuperposition:
    def random_theta(self) -> float:
        return random.random() * 2 * np.pi

    def random_phi(self) -> float:
        return random.random() * np.pi

    def test_superposition(self) -> None:
        n_qubits = 2

        state = ComputationalBasisState(n_qubits)
        state_gates, state_vec = get_random_gates_and_vector()
        for gate in state_gates:
            state = state.with_pauli_gate_applied(gate)

        other_state = ComputationalBasisState(n_qubits)
        other_state_gates, other_state_vec = get_random_gates_and_vector()
        for gate in other_state_gates:
            other_state = other_state.with_pauli_gate_applied(gate)

        theta = self.random_theta()
        phi = self.random_phi()

        sp_state = comp_basis_superposition(state, other_state, theta, phi)

        expected = (
            np.cos(theta) * state_vec
            + np.exp(1j * phi) * np.sin(theta) * other_state_vec
        )
        expected /= np.sqrt(np.sum(np.abs(expected) ** 2))
        actual = raw_vector(sp_state)
        assert abs(actual.conj() @ expected) == pytest.approx(1)

    def test_superposition_when_x_is_equal_to_y(self) -> None:
        n_qubits = 2
        x = random.randint(0, 2**n_qubits - 1)
        y = x
        theta = self.random_theta()
        phi = self.random_phi()
        state_a = ComputationalBasisState(n_qubits, bits=x)
        state_b = ComputationalBasisState(n_qubits, bits=y)
        qs = comp_basis_superposition(state_a, state_b, theta, phi)

        expected = np.zeros(2**n_qubits)
        expected[x] = 1
        actual = raw_vector(qs)
        assert abs(actual.conj() @ expected) == pytest.approx(1)

    def test_superposition_raises_error_when_qubit_counts_are_different(self) -> None:
        with pytest.raises(ValueError):
            comp_basis_superposition(
                ComputationalBasisState(2), ComputationalBasisState(3), 0, 0
            )
