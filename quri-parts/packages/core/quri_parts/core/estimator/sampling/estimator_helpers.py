# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import Callable

from typing_extensions import TypeAlias

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.measurement import CommutablePauliSetMeasurement
from quri_parts.core.operator import CommutablePauliSet, Operator
from quri_parts.core.sampling import PauliSamplingShotsAllocator
from quri_parts.core.state import CircuitQuantumState

#: A function that returns the sequence of (circuit, shot) pairs for performing
#: sampling estimation on the given state. The default operation is that
#: it concatenates the measurement circuits determined by the grouping scheme
#: to the circuit that prepares the state. This is done with the
#: `circuit_shot_pairs_preparation_fn` below. Users may customize this function if
#: additional circuit operations needs to be done other than simple concatenation.
CircuitShotPairPreparationFunction: TypeAlias = Callable[
    [
        CircuitQuantumState,
        Iterable[CommutablePauliSetMeasurement],
        dict[CommutablePauliSet, int],
    ],
    Iterable[tuple[ImmutableQuantumCircuit, int]],
]


def distribute_shots_among_pauli_sets(
    operator: Operator,
    measurement_groups: Iterable[CommutablePauliSetMeasurement],
    shots_allocator: PauliSamplingShotsAllocator,
    total_shots: int,
) -> dict[CommutablePauliSet, int]:
    """Distribute shots to each commuting pauli sets.

    Args:
        operator: The operator to be measured.
        measurement_groups: Sequence of :class:`~CommutablePauliSetMeasurement` that
            corresponds to the grouping result of the operator.
        shot_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
        total_shots: Total number of shots available for sampling measurements.
    """
    pauli_sets = {m.pauli_set for m in measurement_groups}
    shot_allocs = shots_allocator(operator, pauli_sets, total_shots)
    return {pauli_set: n_shots for pauli_set, n_shots in shot_allocs}


def get_sampling_circuits_and_shots(
    state: CircuitQuantumState,
    measurement_groups: Iterable[CommutablePauliSetMeasurement],
    shots_map: dict[CommutablePauliSet, int],
) -> Iterable[tuple[ImmutableQuantumCircuit, int]]:
    """Sets up the (circuit, shot) pairs for performing sampling estimation.
    The circuit is given by the measurement circuit concatenated after the
    circuit held inside the state.

    Args:
        state: The state on which the expectation value is estimated.
        measurement_groups: Sequence of :class:`~CommutablePauliSetMeasurement` that
            corresponds to the grouping result of the operator.
        shots_map: A dictionary whose key is the commuting pauli set and the value is
            the shot count assigned to the commuting pauli set.
    """
    return [
        (
            state.circuit + m.measurement_circuit,
            n_shots,
        )
        for m in measurement_groups
        if (n_shots := shots_map[m.pauli_set]) > 0
    ]
