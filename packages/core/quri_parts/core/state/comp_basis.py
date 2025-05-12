# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cached_property
from typing import Literal, Mapping, Union

import numpy as np
from typing_extensions import TypeAlias

from quri_parts.circuit import (
    GateSequence,
    ImmutableQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    X,
    gate_names,
)
from quri_parts.circuit.gate_names import is_pauli_name

from ..utils.bit import different_bit_index, get_bit
from .state import CircuitQuantumState, GeneralCircuitQuantumState

_SinglePauliNameType: TypeAlias = Union[Literal["X"], Literal["Y"], Literal["Z"]]
_PAULI_NAME_TABLE: dict[int, _SinglePauliNameType] = {1: "X", 2: "Y", 3: "Z"}


def _add_single_pauli(
    state: tuple[int, int, int], pauli_name: _SinglePauliNameType, index: int
) -> tuple[int, int, int]:
    qubit_count, bits, phase = state
    if index >= qubit_count:
        raise ValueError("Index out of range.")

    is_one = get_bit(bits, index)
    if pauli_name in ("X", "Y"):
        bits ^= 1 << index
    if pauli_name == "Y":
        if is_one:
            phase += -1  # -π/2
        else:
            phase += 1  # +π/2
    if is_one and pauli_name == "Z":
        phase += 2  # +π

    return qubit_count, bits, phase


def _add_pauli(state: tuple[int, int, int], gate: QuantumGate) -> tuple[int, int, int]:
    if not is_pauli_name(gate.name):
        raise ValueError(f"Unexpected gate: {gate}. Only a Pauli gate is allowed.")

    if gate.name == gate_names.Pauli:  # multi-qubit Pauli
        for index, pauli_id in zip(gate.target_indices, gate.pauli_ids):
            state = _add_single_pauli(state, _PAULI_NAME_TABLE[pauli_id], index)
    else:  # X, Y, Z
        index = gate.target_indices[0]
        state = _add_single_pauli(state, gate.name, index)

    return state


class ComputationalBasisState(CircuitQuantumState):
    r"""ComputationalBasisState represents a computational basis state. A
    computational basis state can also be considered as a state given as a
    result of applying Pauli gates to \|00...0> state. It internally holds a
    phase factor resulted from the applications of Pauli gates.

    Args:
        n_qubits: The number of qubits.
        bits: An integer representing a bit string of the computational basis state.
    """

    def __init__(self, n_qubits: int, *, bits: int = 0) -> None:
        if bits < 0 or bits >= 2**n_qubits:
            raise ValueError(f"Bits {bits} out of range for {n_qubits}-qubit state")

        self._n_qubits: int = n_qubits
        self._bits: int = bits
        self._phase: int = (
            0  # If _phase is 3, the phase factor will be exp(1j * 3 * π/2).
        )

    def __repr__(self) -> str:
        return "{}(qubit_count={}, bits={}, phase={}π/2)".format(
            self.__class__.__name__,
            self.qubit_count,
            bin(self.bits),
            self._phase,
        )

    def _as_tuple(self) -> tuple[int, int, int]:
        return (self._n_qubits, self._bits, self._phase)

    @staticmethod
    def _from_tuple(state: tuple[int, int, int]) -> "ComputationalBasisState":
        n_qubits, bits, phase = state
        s = ComputationalBasisState(n_qubits, bits=bits)
        s._phase = phase
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComputationalBasisState):
            return False
        return self._as_tuple() == other._as_tuple()

    def __hash__(self) -> int:
        return hash(self._as_tuple())

    @property
    def qubit_count(self) -> int:
        return self._n_qubits

    @cached_property
    def circuit(self) -> ImmutableQuantumCircuit:
        """Circuit to build the quantum state."""
        qc = QuantumCircuit(self._n_qubits)
        for index in range(self._n_qubits):
            if get_bit(self._bits, index):
                qc.add_gate(X(index))
        return qc.freeze()

    def with_pauli_gate_applied(self, gate: QuantumGate) -> "ComputationalBasisState":
        """Apply a Pauli gate to the quantum state."""
        t = _add_pauli(self._as_tuple(), gate)
        return ComputationalBasisState._from_tuple(t)

    def with_gates_applied(
        self, gates: GateSequence
    ) -> Union["ComputationalBasisState", "GeneralCircuitQuantumState"]:
        gate_seq = gates.gates if isinstance(gates, ImmutableQuantumCircuit) else gates
        if all(is_pauli_name(gate.name) for gate in gate_seq):
            state = self._as_tuple()
            for gate in gate_seq:
                state = _add_pauli(state, gate)
            return ComputationalBasisState._from_tuple(state)
        else:
            circuit = self.circuit + gates
            return GeneralCircuitQuantumState(self.qubit_count, circuit)

    @property
    def bits(self) -> int:
        """An integer representing a bit string of the computational basis
        state."""
        return self._bits

    @property
    def phase(self) -> float:
        """The phase of the state."""
        return self._phase * np.pi / 2

    def sample(self, n_shots: int) -> Mapping[int, Union[int, float]]:
        """Sample the state using qulacs."""
        return self.circuit.sample(n_shots)


def comp_basis_superposition(
    state_a: ComputationalBasisState,
    state_b: ComputationalBasisState,
    theta: float,
    phi: float,
) -> GeneralCircuitQuantumState:
    r"""Return a superposition state (as GeneralCircuitQuantumState) composed of
    two ComputationalBasisState.

    .. math::
        \cos \theta | state_a \rangle + e^{i \phi} \sin \theta | state_b \rangle

    Raises ValueError if the qubit counts of the two states are different.
    """
    if state_a.qubit_count != state_b.qubit_count:
        raise ValueError(
            "Superposition of states with different qubit counts is not possible."
        )

    n_qubits = state_a.qubit_count
    phi += state_b.phase - state_a.phase
    x = state_a.bits
    y = state_b.bits

    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        if get_bit(x, i):
            circuit.add_X_gate(i)

    if x == y:
        return GeneralCircuitQuantumState(n_qubits, circuit)

    rotation_targets = []
    xor = x ^ y
    for i in range(n_qubits):
        if get_bit(xor, i):
            rotation_targets.append(i)
    circuit.add_PauliRotation_gate(
        rotation_targets, [1 for _ in rotation_targets], -2 * theta
    )

    diff_bit_index = different_bit_index(x, y)
    if get_bit(y, diff_bit_index):
        sign = 1
    else:
        sign = -1

    angle = 2 * sign * (0.5 * phi - 0.25 * np.pi)
    circuit.add_RZ_gate(diff_bit_index, angle)

    return GeneralCircuitQuantumState(n_qubits, circuit)
