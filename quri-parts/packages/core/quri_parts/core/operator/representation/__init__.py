# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Mapping, Sequence

from typing_extensions import TypeAlias

from ...utils.bit import parity_sign_of_bits
from .bsf import BinarySymplecticVector, bsv_bitwise_commute, pauli_label_to_bsv

if TYPE_CHECKING:
    from ..operator import Operator


#: _TransitionAmplitudeRepresentation represents a operator as a dict whose keys are
#: x of bsv :class:`BinarySymplecticVector` and values are the tuples of the
#: coefficient of the operator with the bsv's phase and z of bsv.
#: This is used only for calculating transition amplitudes efficiently.
_TransitionAmplitudeRepresentation: TypeAlias = Mapping[
    int, Sequence[tuple[complex, int]]
]


def transition_amp_representation(
    operator: "Operator",
) -> _TransitionAmplitudeRepresentation:
    """Returns a binary representation
    :class:`_TransitionAmplitudeRepresentation`, which is the special
    representation designed for calculating tansition amplitudes
    efficiently."""
    ret = defaultdict(list)
    for pauli, coef in operator.items():
        bsv = pauli_label_to_bsv(pauli)
        ret[bsv.x].append((coef * bsv.phase, bsv.z))
    return ret


def transition_amp_comp_basis(
    op_binary_repr: _TransitionAmplitudeRepresentation, m: int, n: int
) -> complex:
    """Returns the transition amplitude :math:`\\langle m|O|n\\rangle` of the operator
    :math:`O`"""
    xb = m ^ n
    if xb not in op_binary_repr:
        return 0

    v: complex = 0
    for coef, zb in op_binary_repr[xb]:
        sign = parity_sign_of_bits(zb & m)
        v += sign * coef
    return v


__all__ = [
    "BinarySymplecticVector",
    "pauli_label_to_bsv",
    "bsv_bitwise_commute",
    "transition_amp_comp_basis",
    "transition_amp_representation",
]
