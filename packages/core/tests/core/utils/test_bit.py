# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.core.utils.bit import (
    bit_length,
    different_bit_index,
    get_bit,
    lowest_bit_index,
    parity_sign_of_bits,
)


def test_get_bit() -> None:
    assert get_bit(3, 0) is True
    assert get_bit(3, 1) is True
    assert get_bit(3, 2) is False


def test_different_bit_index() -> None:
    assert different_bit_index(0b1011, 0b0011) == 3
    assert different_bit_index(0b1000, 0b0011) == 0
    assert different_bit_index(0b1011, 0b0001) == 1


def test_lowest_bit_index() -> None:
    assert lowest_bit_index(0b1011) == 0
    assert lowest_bit_index(0b1000) == 3


def test_parity_sign_of_bits() -> None:
    assert parity_sign_of_bits(0b1011) == -1
    assert parity_sign_of_bits(0b1010) == 1


def test_bit_length() -> None:
    assert bit_length(10) == 4
    assert bit_length(np.int8(10)) == 4
    assert bit_length(np.int16(10)) == 4
    assert bit_length(np.int32(10)) == 4
    assert bit_length(np.int64(10)) == 4
