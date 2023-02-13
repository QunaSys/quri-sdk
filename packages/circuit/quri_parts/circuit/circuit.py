# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
from typing import Optional, Protocol, Union

from typing_extensions import TypeAlias

from .gate import ParametricQuantumGate, QuantumGate
from .gates import (
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
    S,
    Sdag,
    SingleQubitUnitaryMatrix,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    TwoQubitUnitaryMatrix,
    UnitaryMatrix,
    X,
    Y,
    Z,
)

GateSequence: TypeAlias = Union["NonParametricQuantumCircuit", Sequence[QuantumGate]]


class QuantumCircuitProtocol(Protocol):
    """Interface protocol for a quantum circuit.

    This interface covers all quantum circuit classes, including:

    * Non-parametric circuit, parametric circuit and linearly mapped parametric circuit
    * Mutable and immutable circuit classes
    """

    @abstractproperty
    def qubit_count(self) -> int:
        """Number of qubits involved in the circuit."""
        ...

    @abstractproperty
    def depth(self) -> int:
        """Returns circuit depth."""
        ...


class MutableQuantumCircuitProtocol(QuantumCircuitProtocol, Protocol):
    r"""Interface protocol for a mutable quantum circuit.

    This interface represents quantum circuits that can be modified by adding
    :class:`~QuantumGate`\ s (non-parametric gates).

    Some methods (``add_???_gate``) have implementations using an abstract method
    :meth:`~add_gate`.
    """

    @abstractmethod
    def add_gate(self, gate: QuantumGate, gate_index: Optional[int] = None) -> None:
        """Add a (non-parametric) quantum gate to the circuit."""
        ...

    @abstractmethod
    def extend(self, gates: GateSequence) -> None:
        """Extend the circuit with given gate sequence."""
        ...

    def add_Identity_gate(self, qubit_index: int) -> None:
        """Add an Identity gate to the circuit."""
        self.add_gate(Identity(qubit_index))

    def add_X_gate(self, qubit_index: int) -> None:
        """Add an X gate to the circuit."""
        self.add_gate(X(qubit_index))

    def add_Y_gate(self, qubit_index: int) -> None:
        """Add a Y gate to the circuit."""
        self.add_gate(Y(qubit_index))

    def add_Z_gate(self, qubit_index: int) -> None:
        """Add a Z gate to the circuit."""
        self.add_gate(Z(qubit_index))

    def add_H_gate(self, qubit_index: int) -> None:
        """Add an H gate to the circuit."""
        self.add_gate(H(qubit_index))

    def add_S_gate(self, index: int) -> None:
        """Add a S gate to the circuit."""
        self.add_gate(S(index))

    def add_Sdag_gate(self, index: int) -> None:
        """Add a Sdag gate to the circuit."""
        self.add_gate(Sdag(index))

    def add_SqrtX_gate(self, index: int) -> None:
        """Add a SqrtX gate to the circuit."""
        self.add_gate(SqrtX(index))

    def add_SqrtXdag_gate(self, index: int) -> None:
        """Add a SqrtXdag gate to the circuit."""
        self.add_gate(SqrtXdag(index))

    def add_SqrtY_gate(self, index: int) -> None:
        """Add a SqrtY gate to the circuit."""
        self.add_gate(SqrtY(index))

    def add_SqrtYdag_gate(self, index: int) -> None:
        """Add a SqrtYdag gate to the circuit."""
        self.add_gate(SqrtYdag(index))

    def add_T_gate(self, index: int) -> None:
        """Add a T gate to the circuit."""
        self.add_gate(T(index))

    def add_Tdag_gate(self, index: int) -> None:
        """Add a Tdag gate to the circuit."""
        self.add_gate(Tdag(index))

    def add_U1_gate(self, index: int, lmd: float) -> None:
        """Add an U1 gate to the circuit."""
        self.add_gate(U1(index, lmd))

    def add_U2_gate(self, index: int, phi: float, lmd: float) -> None:
        """Add an U2 gate to the circuit."""
        self.add_gate(U2(index, phi, lmd))

    def add_U3_gate(self, index: int, theta: float, phi: float, lmd: float) -> None:
        """Add an U3 gate to the circuit."""
        self.add_gate(U3(index, theta, phi, lmd))

    def add_RX_gate(self, index: int, angle: float) -> None:
        """Add a RX gate to the circuit."""
        self.add_gate(RX(index, angle))

    def add_RY_gate(self, index: int, angle: float) -> None:
        """Add a RY gate to the circuit."""
        self.add_gate(RY(index, angle))

    def add_RZ_gate(self, index: int, angle: float) -> None:
        """Add a RZ gate to the circuit."""
        self.add_gate(RZ(index, angle))

    def add_CNOT_gate(self, control_index: int, target_index: int) -> None:
        """Add a CNOT gate to the circuit."""
        self.add_gate(CNOT(control_index, target_index))

    def add_CZ_gate(self, control_qubit_index: int, target_qubit_index: int) -> None:
        """Add a Control-Z gate to the circuit."""
        self.add_gate(CZ(control_qubit_index, target_qubit_index))

    def add_SWAP_gate(self, target_index1: int, target_index2: int) -> None:
        """Add a SWAP gate to the circuit."""
        self.add_gate(SWAP(target_index1, target_index2))

    def add_TOFFOLI_gate(
        self, control_index1: int, control_index2: int, target_index: int
    ) -> None:
        """Add a TOFFOLI gate to the circuit."""
        self.add_gate(TOFFOLI(control_index1, control_index2, target_index))

    def add_UnitaryMatrix_gate(
        self,
        target_indices: Sequence[int],
        unitary_matrix: Sequence[Sequence[complex]],
    ) -> None:
        """Add a UnitaryMatrix gate to the circuit."""
        self.add_gate(UnitaryMatrix(target_indices, unitary_matrix))

    def add_SingleQubitUnitaryMatrix_gate(
        self,
        target_index: int,
        unitary_matrix: Sequence[Sequence[complex]],
    ) -> None:
        """Add a single qubit UnitaryMatrix gate to the circuit."""
        self.add_gate(SingleQubitUnitaryMatrix(target_index, unitary_matrix))

    def add_TwoQubitUnitaryMatrix_gate(
        self,
        target_index1: int,
        target_index2: int,
        unitary_matrix: Sequence[Sequence[complex]],
    ) -> None:
        """Add a two qubit UnitaryMatrix gate to the circuit."""
        self.add_gate(
            TwoQubitUnitaryMatrix(target_index1, target_index2, unitary_matrix)
        )

    def add_Pauli_gate(
        self, target_indices: Sequence[int], pauli_ids: Sequence[int]
    ) -> None:
        """Add a Pauli gate to the circuit."""
        self.add_gate(Pauli(target_indices, pauli_ids))

    def add_PauliRotation_gate(
        self,
        target_qubits: Sequence[int],
        pauli_id_list: Sequence[int],
        angle: float,
    ) -> None:
        """Add a Pauli rotation gate to the circuit."""
        self.add_gate(PauliRotation(target_qubits, pauli_id_list, angle))


class NonParametricQuantumCircuit(QuantumCircuitProtocol, ABC):
    """A base class for quantum circuits having only non-parametric gates.

    This class support ``+`` operator with ``GateSequence``.
    """

    @abstractproperty
    def gates(self) -> Sequence[QuantumGate]:
        """Returns the gate sequence of the circuit."""
        ...

    @property
    def depth(self) -> int:
        max_layers_each_qubit = [0] * self.qubit_count
        for gate in self.gates:
            acting_ids = [*gate.control_indices, *gate.target_indices]
            max_layers = max(max_layers_each_qubit[index] for index in acting_ids)
            for index in acting_ids:
                max_layers_each_qubit[index] = max_layers + 1
        return max(max_layers_each_qubit)

    def combine(self, gates: GateSequence) -> "QuantumCircuit":
        """Create a new circuit with itself and the given gates combined."""
        circuit = QuantumCircuit(self.qubit_count)
        circuit.extend(self)
        circuit.extend(gates)
        return circuit

    @abstractmethod
    def freeze(self) -> "ImmutableQuantumCircuit":
        """Returns a \"freezed\" version of itself.

        The \"freezed\" version is an immutable object and can be reused
        safely without copying.
        """
        ...

    @abstractmethod
    def get_mutable_copy(self) -> "QuantumCircuit":
        """Returns a copy of itself that can be modified.

        Use this method when you want to get a new circuit based on an
        existing circuit but don't want to modify it.
        """
        ...

    def __add__(self, gates: GateSequence) -> "QuantumCircuit":
        return self.combine(gates)


class QuantumCircuit(NonParametricQuantumCircuit, MutableQuantumCircuitProtocol):
    """A mutable quantum circuit having only non-parametric gates."""

    def __init__(
        self,
        qubit_count: int,
        gates: Sequence[QuantumGate] = [],
    ):
        self._qubit_count = qubit_count
        self._gates = list(gates)

    @property
    def qubit_count(self) -> int:
        return self._qubit_count

    @property
    def gates(self) -> Sequence[QuantumGate]:
        return tuple(self._gates)

    def freeze(self) -> "ImmutableQuantumCircuit":
        return ImmutableQuantumCircuit(self)

    def get_mutable_copy(self) -> "QuantumCircuit":
        circuit = QuantumCircuit(self.qubit_count, self._gates)
        return circuit

    def add_gate(self, gate: QuantumGate, gate_index: Optional[int] = None) -> None:
        if isinstance(gate, ParametricQuantumGate):
            raise ValueError(
                "QuantumCircuit only accepts QuantumGate, not ParametricQuantumGate."
            )
        if max([*gate.control_indices, *gate.target_indices]) >= self.qubit_count:
            raise ValueError(
                "The indices of the gate applied must be smaller than qubit_count"
            )
        if gate_index:
            self._gates.insert(gate_index, gate)
        else:
            self._gates.append(gate)

    def extend(self, gates: GateSequence) -> None:
        if isinstance(gates, NonParametricQuantumCircuit):
            gates = gates.gates
        for gate in gates:
            self.add_gate(gate)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NonParametricQuantumCircuit):
            return False
        return self.freeze() == other.freeze()

    def __iadd__(self, gates: GateSequence) -> "QuantumCircuit":
        self.extend(gates)
        return self


class ImmutableQuantumCircuit(NonParametricQuantumCircuit):
    """An immutable quantum circuit having only non-parametric gates."""

    def __init__(self, circuit: QuantumCircuit):
        self._qubit_count = circuit.qubit_count
        self._gates = circuit.gates

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NonParametricQuantumCircuit):
            return False
        f = other.freeze()
        return (self.qubit_count, self.gates) == (f.qubit_count, f.gates)

    @property
    def qubit_count(self) -> int:
        return self._qubit_count

    @property
    def gates(self) -> Sequence[QuantumGate]:
        return self._gates

    def freeze(self) -> "ImmutableQuantumCircuit":
        return self

    def get_mutable_copy(self) -> QuantumCircuit:
        return QuantumCircuit(self.qubit_count, self.gates)
