# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple

from ..pauli import PauliLabel, SinglePauli


class BinarySymplecticVector(NamedTuple):
    """Binary symplectic vector representation for Pauli operators."""

    x: int
    z: int
    phase: complex = 1


def pauli_label_to_bsv(pauli: PauliLabel) -> BinarySymplecticVector:
    """Represent a PauliLabel as a binary symplectic vector."""
    x: int = 0
    z: int = 0
    phase: complex = 1
    for i, p in pauli:
        b = 1 << i
        if p == SinglePauli.X:
            x += b
        elif p == SinglePauli.Y:
            x += b
            z += b
            phase *= -1j
        elif p == SinglePauli.Z:
            z += b
    return BinarySymplecticVector(x=x, z=z, phase=phase)


def bsv_bitwise_commute(
    bsv1: BinarySymplecticVector, bsv2: BinarySymplecticVector
) -> bool:
    """Return if two Pauli strings are bitwise commuting."""
    x1_z2 = bsv1.x & bsv2.z
    z1_x2 = bsv1.z & bsv2.x
    non_commuting = x1_z2 ^ z1_x2
    return non_commuting == 0
