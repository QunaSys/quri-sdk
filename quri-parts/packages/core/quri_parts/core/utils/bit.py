# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np


def bit_length(bits: Union[int, np.int8, np.int16, np.int32, np.int64]) -> int:
    """Computes the bit length of an integer."""
    return int(bits).bit_length()


def get_bit(x: int, index: int) -> bool:
    """Returns if the bit at 'index' is set or not."""
    mask = 1 << index
    return (x & mask) != 0


def different_bit_index(x: int, y: int) -> int:
    """Returns the index of the lowest different bit."""
    return lowest_bit_index(x ^ y)


def lowest_bit_index(x: int) -> int:
    """Returns the index of the lowest bit that is set."""
    if x == 0:
        raise ValueError("Given integer is zero.")
    for i in range(64):
        mask = 1 << i
        if (x & mask) != 0:
            return i
    raise ValueError("Given integer is too large.")


def parity_sign_of_bits(bits: int) -> int:
    """Returns a sign corresponding to parity of bits (even=1, odd=-1)."""
    return 1 - 2 * (bin(bits).count("1") % 2)
