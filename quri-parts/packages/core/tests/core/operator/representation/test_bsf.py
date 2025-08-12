# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.core.operator import pauli_label
from quri_parts.core.operator.representation import (
    BinarySymplecticVector,
    bsv_bitwise_commute,
    pauli_label_to_bsv,
)


def test_pauli_label_to_bsv() -> None:
    pauli = pauli_label("X2 Y4 Z7")
    bsv = pauli_label_to_bsv(pauli)

    assert bsv.x == 0b00010100
    assert bsv.z == 0b10010000


def test_bsv_bitwise_commute() -> None:
    bsv1 = BinarySymplecticVector(x=0b101100, z=0b010111)
    bsv2 = BinarySymplecticVector(x=0b001100, z=0b000101)
    assert bsv_bitwise_commute(bsv1, bsv2)

    bsv3 = BinarySymplecticVector(x=0b001100, z=0b000001)
    assert not bsv_bitwise_commute(bsv1, bsv3)

    bsv4 = BinarySymplecticVector(x=0b001101, z=0b000100)
    assert not bsv_bitwise_commute(bsv1, bsv4)
