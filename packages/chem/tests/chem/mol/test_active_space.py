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

from quri_parts.chem.mol import (
    convert_to_spin_orbital_indices,
    get_core_and_active_orbital_indices,
)


class TestCoreAndActiveOrbitals:
    def test_core_and_active_orbital_indices(self) -> None:
        occupied_indices, active_indices = get_core_and_active_orbital_indices(
            n_active_orb=4, n_active_ele=2, n_electrons=10
        )
        assert occupied_indices == [0, 1, 2, 3]
        assert active_indices == [4, 5, 6, 7]

    def test_core_and_active_orbital_indices_with_active_orbs(self) -> None:
        with pytest.raises(ValueError):
            get_core_and_active_orbital_indices(
                n_active_orb=4,
                n_active_ele=2,
                n_electrons=10,
                active_orbs_indices=[1, 2],
            )
        occupied_indices, active_indices = get_core_and_active_orbital_indices(
            n_active_orb=4,
            n_active_ele=2,
            n_electrons=10,
            active_orbs_indices=[2, 3, 5, 6],
        )
        assert occupied_indices == [0, 1, 4, 7]
        assert active_indices == [2, 3, 5, 6]

    def test_core_and_active_orbital_indices_with_interval(self) -> None:
        occupied_indices, active_indices = get_core_and_active_orbital_indices(
            n_active_orb=4,
            n_active_ele=2,
            n_electrons=10,
            active_orbs_indices=[5, 6, 7, 8],
        )
        assert occupied_indices == [0, 1, 2, 3]
        assert active_indices == [5, 6, 7, 8]


def test_convert_to_spin_orbital_indices() -> None:
    assert convert_to_spin_orbital_indices([0, 1], [2, 3]) == (
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    )
    assert convert_to_spin_orbital_indices([0, 3], [7, 9]) == (
        [0, 1, 6, 7],
        [14, 15, 18, 19],
    )
