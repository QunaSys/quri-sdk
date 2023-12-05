# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import NamedTuple, TypeVar

from .gate_names import UnitaryMatrix

_QuantumGateT = TypeVar("_QuantumGateT", "QuantumGate", "ParametricQuantumGate")


class QuantumGate(NamedTuple):
    """Non-parametric quantum gate.

    Not intended for direct use. Every gate is created by factory
    methods. A QuantumGate object contains information of gate name,
    control qubit, target qubit, classical bits, parameters, and pauli
    ids.
    """

    name: str
    target_indices: Sequence[int]
    control_indices: Sequence[int] = ()
    classical_indices: Sequence[int] = ()
    params: Sequence[float] = ()
    pauli_ids: Sequence[int] = ()
    unitary_matrix: Sequence[Sequence[complex]] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantumGate):
            return False
        return is_gate_equal(self, other)


class ParametricQuantumGate(NamedTuple):
    """Parametric quantum gate.

    Not intended for direct use. Every gate is created through factory
    methods. A ParametricQuantumGate object contains information of gate
    name, control qubit, target qubit, and pauli ids.
    """

    name: str
    target_indices: Sequence[int]
    control_indices: Sequence[int] = ()
    pauli_ids: Sequence[int] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParametricQuantumGate):
            return False
        return is_gate_equal(self, other)


def is_gate_equal(
    gate1: _QuantumGateT,
    gate2: _QuantumGateT,
) -> bool:
    if gate1.name != gate2.name:
        return False
    if set(gate1.control_indices) != set(gate1.control_indices):
        return False
    if gate1.name == UnitaryMatrix:
        if gate1.target_indices != gate2.target_indices:
            return False
    else:
        if set(gate1.target_indices) != set(gate2.target_indices):
            return False
    if set(zip(gate1.target_indices, gate1.pauli_ids)) != set(
        zip(gate2.target_indices, gate2.pauli_ids)
    ):
        return False
    if isinstance(gate1, QuantumGate):
        if gate1.params != gate2.params:
            return False
        if gate1.name == "Measurement":
            return set(zip(gate1.target_indices, gate1.classical_indices)) == set(
                zip(gate2.target_indices, gate2.classical_indices)
            )
        return gate1.unitary_matrix == gate2.unitary_matrix
    return True
