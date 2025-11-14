# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod, abstractproperty
from collections.abc import Sequence
from typing import Optional, Protocol, Union

from typing_extensions import TypeAlias, TypeGuard

#: An immutable quantum circuit having only non-parametric gates.
#: A mutable quantum circuit having only non-parametric gates.
from quri_parts.rust.circuit.circuit import ImmutableQuantumCircuit, QuantumCircuit

from .gate import QuantumGate
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
    Measurement,
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

GateSequence: TypeAlias = Union["ImmutableQuantumCircuit", Sequence[QuantumGate]]


def is_gate_sequence(
    gates: Union["QuantumCircuitProtocol", GateSequence]
) -> TypeGuard[GateSequence]:
    return isinstance(gates, (ImmutableQuantumCircuit, Sequence))


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
    def cbit_count(self) -> int:
        """Number of classical bits involved in the circuit."""
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

    def measure(
        self,
        qubit_indices: Union[int, Sequence[int]],
        classical_indices: Union[int, Sequence[int]],
    ) -> None:
        """Adds measurement gate at selected qubits."""
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]
        if isinstance(classical_indices, int):
            classical_indices = [classical_indices]

        self.add_gate(Measurement(qubit_indices, classical_indices))


#: A base class for quantum circuits having only non-parametric gates.
#:
#: This class support ``+`` operator with ``GateSequence``.
ImmutableQuantumCircuit = ImmutableQuantumCircuit


#: Deprecated: use `quri_parts.circuit.ImmutableQuantumCircuit` instead
NonParametricQuantumCircuit = ImmutableQuantumCircuit

__all__ = [
    "GateSequence",
    "ImmutableQuantumCircuit",
    "MutableQuantumCircuitProtocol",
    "NonParametricQuantumCircuit",
    "QuantumCircuit",
    "QuantumCircuitProtocol",
]
