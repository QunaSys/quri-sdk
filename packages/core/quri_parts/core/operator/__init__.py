# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .operator import Operator, commutator, is_hermitian, is_ops_close, truncate, zero
from .pauli import (
    PAULI_IDENTITY,
    CommutablePauliSet,
    PauliLabel,
    SinglePauli,
    pauli_label,
    pauli_name,
    pauli_product,
)
from .representation import transition_amp_comp_basis, transition_amp_representation
from .trotter_suzuki import trotter_suzuki_decomposition

PAULI_IDENTITY = PAULI_IDENTITY
"""PauliLabel used as an identity."""

#: CommutablePauliSet
CommutablePauliSet = CommutablePauliSet

__all__ = [
    "CommutablePauliSet",
    "commutator",
    "truncate",
    "is_hermitian",
    "is_ops_close",
    "PAULI_IDENTITY",
    "pauli_label",
    "pauli_name",
    "pauli_product",
    "PauliLabel",
    "Operator",
    "SinglePauli",
    "transition_amp_comp_basis",
    "transition_amp_representation",
    "zero",
    "trotter_suzuki_decomposition",
]
