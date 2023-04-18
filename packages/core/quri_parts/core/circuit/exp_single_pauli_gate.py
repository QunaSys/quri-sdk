# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import PauliRotation, QuantumGate
from quri_parts.core.operator import PauliLabel


def convert_exp_single_pauli_gate(pauli: PauliLabel, coef: float) -> QuantumGate:
    r"""Convert an exponentiated single Pauli :math:`\exp(i a P)` to the
    PauliRotation gate, where :math:`a` is a real number coefficient and
    :math:`P` is a product of the Pauli.

    Args:
        pauli: :class:`PauliLabel` for the exponentiated single Pauli.
        coef: A real number that is a coefficient of an exponentiated single Pauli.
    """
    target_indices, pauli_ids = zip(*pauli)
    c = -1 * coef
    return PauliRotation(target_indices, pauli_ids, 2 * c)
