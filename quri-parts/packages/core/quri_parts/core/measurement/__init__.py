# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Union

from quri_parts.core.operator import Operator, PauliLabel

from .bitwise_commuting_pauli import (
    bitwise_commuting_pauli_measurement,
    bitwise_commuting_pauli_measurement_circuit,
    bitwise_pauli_reconstructor_factory,
    individual_pauli_measurement,
)
from .interface import (
    CommutablePauliSetMeasurement,
    CommutablePauliSetMeasurementFactory,
    CommutablePauliSetMeasurementTuple,
    PauliMeasurementCircuitGeneration,
    PauliReconstructor,
    PauliReconstructorFactory,
)

#: PauliMeasurementCircuitGeneration represents a function that generates a circuit
#: (a gate list) for measuring mutually commuting Pauli operators.
PauliMeasurementCircuitGeneration = PauliMeasurementCircuitGeneration

#: PauliReconstructor represents a function that reconstructs a value of a Pauli
#:  operator from a measurement result of its measurement circuit.
PauliReconstructor = PauliReconstructor

#: PauliReconstructorFactory represents a factory function that returns
#: a :class:`~PauliReconstructor` for a given Pauli operator.
PauliReconstructorFactory = PauliReconstructorFactory

#: Represents a function that performs grouping of Pauli operators into groups of
#: commutable Pauli operators and returns measurement schemes for them.
CommutablePauliSetMeasurementFactory = CommutablePauliSetMeasurementFactory


class CachedMeasurementFactory:
    """A class decorator that converts a
    :class:`CommutablePauliSetMeasurementFactory` to a new
    :class:`CommutablePauliSetMeasurementFactory` runs the same grouping
    algorithm but caches grouping result for later usage.

    Example:
    >>> cached_measurement_factory = CachedMeasuremetFactory(
    ...    bitwise_commuting_pauli_measurement
    ... )
    >>> operator = Operator({
    ...    pauli_label("X0 Y1"): 1,
    ...    pauli_label("X0 Z2"): 2,
    ...    pauli_label("Y0 Z2"): 3,
    ...    PAULI_IDENTITY: 4
    ... })
    >>> cached_measurement_factory(operator)
    """

    def __init__(
        self, measurement_factory: CommutablePauliSetMeasurementFactory
    ) -> None:
        self._measurement_factory = measurement_factory
        self._cache: dict[
            frozenset[tuple[PauliLabel, complex]],
            Iterable[CommutablePauliSetMeasurement],
        ] = {}

    def __call__(
        self, paulis: Union[Operator, Iterable[PauliLabel]]
    ) -> Iterable[CommutablePauliSetMeasurement]:
        if not isinstance(paulis, Operator):
            paulis = Operator({p: 1 + 0j for p in paulis})

        op_key = frozenset(paulis.items())
        if op_key in self._cache:
            return self._cache[op_key]
        groups = self._measurement_factory(paulis)
        self._cache[op_key] = groups
        return groups

    @property
    def cached_groups(
        self,
    ) -> dict[
        frozenset[tuple[PauliLabel, complex]],
        Iterable[CommutablePauliSetMeasurement],
    ]:
        return self._cache.copy()


__all__ = [
    "PauliMeasurementCircuitGeneration",
    "PauliReconstructor",
    "PauliReconstructorFactory",
    "CommutablePauliSetMeasurement",
    "CommutablePauliSetMeasurementTuple",
    "CommutablePauliSetMeasurementFactory",
    "bitwise_commuting_pauli_measurement_circuit",
    "bitwise_pauli_reconstructor_factory",
    "bitwise_commuting_pauli_measurement",
    "individual_pauli_measurement",
    "CachedMeasurementFactory",
]
