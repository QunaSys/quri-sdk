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
from typing import Union

from qulacs import GeneralQuantumOperator
from typing_extensions import TypeAlias

from quri_parts.core.operator import Operator, PauliLabel, pauli_name


def _qulacs_pauli_str(pauli_label: PauliLabel) -> str:
    s = [f"{pauli_name(p)} {i}" for i, p in pauli_label]
    return " ".join(s)


_OperatorKey: TypeAlias = Union[PauliLabel, frozenset[tuple[PauliLabel, complex]]]
_operator_cache: dict[tuple[_OperatorKey, int], GeneralQuantumOperator] = {}


def convert_operator(
    operator: Union[Operator, PauliLabel], n_qubits: int
) -> GeneralQuantumOperator:
    """Convert an :class:`~Operator` or a :class:`~PauliLabel` to a
    :class:`qulacs.GeneralQuantumOperator`."""
    op_key: _OperatorKey
    if isinstance(operator, PauliLabel):
        op_key = frozenset({(operator, 1.0)})
    else:
        op_key = frozenset(operator.items())
    if (op_key, n_qubits) in _operator_cache:
        return _operator_cache[(op_key, n_qubits)]

    op = GeneralQuantumOperator(n_qubits)
    paulis: Iterable[tuple[PauliLabel, complex]]
    if isinstance(operator, Operator):
        paulis = operator.items()
    else:
        paulis = [(operator, 1)]
    for pauli, coef in paulis:
        op.add_operator(coef, _qulacs_pauli_str(pauli))

    _operator_cache[(op_key, n_qubits)] = op
    return op
