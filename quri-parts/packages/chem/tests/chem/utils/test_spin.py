# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.chem.utils.spin import occupation_state_sz, reference_occupation


def test_occupation_state_sz() -> None:
    assert occupation_state_sz([0]) == 0.5
    assert occupation_state_sz([0, 1]) == 0
    assert occupation_state_sz([0, 2]) == 1.0
    assert occupation_state_sz([3, 6]) == 0.0
    assert occupation_state_sz([2, 1]) == 0.0
    assert occupation_state_sz([1, 3]) == -1.0


def test_reference_occupation() -> None:
    assert reference_occupation(8, 4) == [0, 1, 2, 3]
    assert reference_occupation(8, 4, 0) == [0, 1, 2, 3]
    assert reference_occupation(8, 3, 0.5) == [0, 1, 2]
    assert reference_occupation(8, 3, -0.5) == [0, 1, 3]
    assert reference_occupation(8, 3, 1.5) == [0, 2, 4]
    assert reference_occupation(16, 11, 2.5) == [
        0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14
    ]  # fmt: skip
    with pytest.raises(ValueError):
        reference_occupation(8, 3, 0)  # odd count needs a half-integer sz
    with pytest.raises(ValueError):
        reference_occupation(6, 4, 2)  # only 3 spin-up orbitals
    with pytest.raises(ValueError):
        reference_occupation(2, 3)  # more fermions than spin orbitals
    with pytest.raises(ValueError):
        reference_occupation(8, 3, 0.5000000000000001)  # not a half-integer
