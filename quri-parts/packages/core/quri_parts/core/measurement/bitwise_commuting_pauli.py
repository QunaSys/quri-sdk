# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Iterable, Sequence
from typing import Union

from quri_parts.circuit import H, QuantumGate, Sdag
from quri_parts.core.operator import (
    PAULI_IDENTITY,
    CommutablePauliSet,
    Operator,
    PauliLabel,
    SinglePauli,
)
from quri_parts.core.operator.grouping import (
    bitwise_pauli_grouping,
    individual_pauli_grouping,
)
from quri_parts.core.operator.representation import pauli_label_to_bsv
from quri_parts.core.utils.bit import parity_sign_of_bits

from .interface import (
    CommutablePauliSetMeasurement,
    CommutablePauliSetMeasurementTuple,
    PauliReconstructor,
)


def bitwise_commuting_pauli_measurement_circuit(
    pauli_set: CommutablePauliSet,
) -> Sequence[QuantumGate]:
    r"""An implementation of :class:`~PauliMeasurementCircuitGeneration`, which
    generates a circuit (a gate list) for a "trivial" measurement of bitwise-
    commuting Pauli strings.

    The "trivial" measurement of a Pauli string refers to a measurement
    in computational basis preceded by single qubit rotations to map
    each Pauli matrix on each qubit to :math:`Z`. For example, if the
    Pauli string contains :math:`X` (:math:`Y`) at a qubit index
    :math:`i`, then (an :math:`S^\dagger` gate and) an Hadamard gate is
    applied on the qubit :math:`i` before performing the :math:`Z`
    measurements on all qubits.
    """
    if len(pauli_set) == 0:
        raise ValueError("No Pauli operators to measure supplied (pauli_set is empty).")

    pauli_map: dict[int, int] = {}
    for pauli_label in pauli_set:
        for index, pauli in pauli_label:
            if index in pauli_map and pauli_map[index] != pauli:
                raise ValueError(
                    "pauli_set contains pauli operators not commuting at qubit index "
                    f"{index}."
                )
            pauli_map[index] = pauli

    circuit: list[QuantumGate] = []
    for index, pauli in pauli_map.items():
        if pauli == SinglePauli.X:
            circuit.append(H(index))
        elif pauli == SinglePauli.Y:
            circuit.append(Sdag(index))
            circuit.append(H(index))

    # convert to tuple to avoid mutation
    return tuple(circuit)


def bitwise_pauli_reconstructor_factory(pauli: PauliLabel) -> PauliReconstructor:
    r"""A factory of a function that reconstructs a value of the given Pauli
    operator from a result of a "trivial" measurement of bitwise-commuting
    Pauli strings.

    The "trivial" measurement of a Pauli string refers to a measurement
    in computational basis preceded by single qubit rotations to map
    each Pauli matrix on each qubit to :math:`Z`. For example, if the
    Pauli string contains :math:`X` (:math:`Y`) at a qubit index
    :math:`i`, then (an :math:`S^\dagger` gate and) an Hadamard gate is
    applied on the qubit :math:`i` before performing the :math:`Z`
    measurements on all qubits.
    """
    if pauli == PAULI_IDENTITY:
        return lambda bits: 1

    bsv = pauli_label_to_bsv(pauli)
    p_z_or_x = bsv.z | bsv.x
    return lambda bits: parity_sign_of_bits(bits & p_z_or_x)


def bitwise_commuting_pauli_measurement(
    paulis: Union[Operator, Iterable[PauliLabel]]
) -> Collection[CommutablePauliSetMeasurement]:
    """Groups the given Pauli operators into sets of bitwise commuting Pauli
    operators and returns measurement schemes for them."""
    pauli_sets = bitwise_pauli_grouping(paulis)
    return tuple(
        CommutablePauliSetMeasurementTuple(
            pauli_set=pauli_set,
            measurement_circuit=bitwise_commuting_pauli_measurement_circuit(pauli_set),
            pauli_reconstructor_factory=bitwise_pauli_reconstructor_factory,
        )
        for pauli_set in pauli_sets
    )


def individual_pauli_measurement(
    paulis: Union[Operator, Iterable[PauliLabel]]
) -> Collection[CommutablePauliSetMeasurement]:
    """Returns measurement schemes for Pauli operators where each operator is
    measured individually (i.e. no grouping)."""
    pauli_sets = individual_pauli_grouping(paulis)
    return tuple(
        CommutablePauliSetMeasurementTuple(
            pauli_set=pauli_set,
            measurement_circuit=bitwise_commuting_pauli_measurement_circuit(pauli_set),
            pauli_reconstructor_factory=bitwise_pauli_reconstructor_factory,
        )
        for pauli_set in pauli_sets
    )
