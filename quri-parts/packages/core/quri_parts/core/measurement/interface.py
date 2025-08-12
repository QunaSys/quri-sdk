# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import Callable, NamedTuple, Protocol, Union

from typing_extensions import TypeAlias

from quri_parts.circuit import QuantumGate
from quri_parts.core.operator import CommutablePauliSet, Operator, PauliLabel

#: PauliMeasurementCircuitGeneration represents a function that generates a circuit
#: (a gate list) for measuring mutually commuting Pauli operators.
PauliMeasurementCircuitGeneration: TypeAlias = Callable[
    [CommutablePauliSet], Sequence[QuantumGate]
]

#: PauliReconstructor represents a function that reconstructs a value of a Pauli
#: operator from a measurement result of its measurement circuit.
PauliReconstructor: TypeAlias = Callable[[int], int]

#: PauliReconstructorFactory represents a factory function that returns
#: a :class:`~PauliReconstructor` for a given Pauli operator.
PauliReconstructorFactory: TypeAlias = Callable[[PauliLabel], PauliReconstructor]


class CommutablePauliSetMeasurement(Protocol):
    """Represents a measurement scheme for a set of commutable Pauli
    operators."""

    @property
    @abstractmethod
    def pauli_set(self) -> CommutablePauliSet:
        """A set of commutable Pauli operators subject to the measurement."""
        ...

    @property
    @abstractmethod
    def measurement_circuit(self) -> Sequence[QuantumGate]:
        """A circuit required to measure the given commutable Pauli operators
        at once."""
        ...

    @property
    @abstractmethod
    def pauli_reconstructor_factory(self) -> PauliReconstructorFactory:
        """A factory of :class:`~PauliReconstructor` that reconstructs a value
        of a Pauli operator from a measurement result of the measurement
        circuit."""
        ...


class CommutablePauliSetMeasurementTuple(NamedTuple):
    pauli_set: CommutablePauliSet
    measurement_circuit: Sequence[QuantumGate]
    pauli_reconstructor_factory: PauliReconstructorFactory


#: Represents a function that performs grouping of Pauli operators into sets of
#: commutable Pauli operators and returns measurement schemes for them.
CommutablePauliSetMeasurementFactory: TypeAlias = Callable[
    [Union[Operator, Iterable[PauliLabel]]], Iterable[CommutablePauliSetMeasurement]
]
