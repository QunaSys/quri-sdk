# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Sequence, Union

from stim import PauliString
from typing_extensions import TypeAlias

from quri_parts.core.operator import Operator, PauliLabel


def _pauli_indices(pauli_label: PauliLabel, qubit_count: int) -> Sequence[int]:
    pauli_indices = [0] * qubit_count
    max_index = 0
    for index, pauli in pauli_label:
        max_index = max(max_index, index)
        pauli_indices[index] = pauli
    return pauli_indices[: max_index + 1]


_OperatorKey: TypeAlias = frozenset[tuple[PauliLabel, complex]]
_operator_cache: dict[
    tuple[_OperatorKey, int], Sequence[tuple[PauliString, complex]]
] = {}


def convert_operator(
    operator: Union[Operator, PauliLabel], qubit_count: int
) -> Sequence[tuple[PauliString, complex]]:
    """
    Converts an :class:`~Operator` (or :class:`~PauliLabel`) to the Sequence of
    :class:`stim.PauliString` and it's coefficient.
    """
    paulis: Iterable[tuple[PauliLabel, complex]]
    op_key: _OperatorKey
    if isinstance(operator, PauliLabel):
        paulis = [(operator, 1)]
    else:
        paulis = operator.items()

    op_key = frozenset(paulis)
    if (op_key, qubit_count) in _operator_cache:
        return _operator_cache[(op_key, qubit_count)]

    ret = []
    for pauli, coef in paulis:
        pauli_indices = _pauli_indices(pauli, qubit_count)
        ret.append((PauliString(pauli_indices), coef))

    _operator_cache[(op_key, qubit_count)] = ret
    return ret


__all__ = ["convert_operator"]
