# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.openfermion.utils.post_selection_filters import (
    create_bk_electron_number_post_selection_filter_fn,
    create_jw_electron_number_post_selection_filter_fn,
    create_scbk_electron_number_post_selection_filter_fn,
)


def test_create_jw_electron_number_post_selection_filter_fn() -> None:
    n_electrons = 3

    filter_fn = create_jw_electron_number_post_selection_filter_fn(n_electrons)
    assert not filter_fn(0)
    assert not filter_fn(0b01)
    assert not filter_fn(0b11)
    assert filter_fn(0b101010)
    assert filter_fn(7)

    filter_fn = create_jw_electron_number_post_selection_filter_fn(n_electrons, sz=0.5)
    assert not filter_fn(0)
    assert not filter_fn(0b01)
    assert filter_fn(0b111)
    assert filter_fn(0b11100)
    assert filter_fn(193)
    assert not filter_fn(0b1011)
    assert not filter_fn(0b11010)

    n_electrons = 2
    filter_fn = create_jw_electron_number_post_selection_filter_fn(n_electrons, sz=0)
    assert not filter_fn(0b00)
    assert not filter_fn(0b10)
    assert filter_fn(0b11)
    assert filter_fn(0b1100)
    assert filter_fn(0b1001)
    assert not filter_fn(0b101)
    assert not filter_fn(0b1111)


def test_create_bk_electron_number_post_selection_filter_fn() -> None:
    qubit_count = 10
    n_electrons = 2

    filter_fn = create_bk_electron_number_post_selection_filter_fn(
        qubit_count, n_electrons
    )
    assert not filter_fn(0)
    assert filter_fn(0b01)
    assert filter_fn(0b110)
    assert not filter_fn(0b101)
    assert filter_fn(11)

    filter_fn = create_bk_electron_number_post_selection_filter_fn(
        qubit_count, n_electrons, sz=1.0
    )
    assert not filter_fn(0)
    assert not filter_fn(0b01)
    assert filter_fn(0b0111)
    assert not filter_fn(0b0010)
    assert filter_fn(59)

    filter_fn = create_bk_electron_number_post_selection_filter_fn(
        qubit_count, n_electrons, sz=0.0
    )
    assert filter_fn(0b1)
    assert filter_fn(0b11)
    assert filter_fn(0b101011)
    assert filter_fn(0b1011)
    assert filter_fn(0b1100)
    assert filter_fn(0b100)
    assert not filter_fn(0b111)
    assert not filter_fn(0b111011)
    assert not filter_fn(0b1001011)
    assert not filter_fn(0b1000)
    assert not filter_fn(0b1000010)
    assert not filter_fn(0b1110000100)


def test_create_scbk_electron_number_post_selection_filter_fn() -> None:
    qubit_count = 6
    n_electrons = 2

    filter_fn = create_scbk_electron_number_post_selection_filter_fn(
        qubit_count, n_electrons, sz=0.0
    )
    assert filter_fn(0b11011)  # [0, 1]
    assert not filter_fn(0b1)  # [0, 2]
    assert not filter_fn(0b1000)  # [1, 3]
    assert filter_fn(0)  # [6, 7]

    # three electrons
    filter_fn = create_scbk_electron_number_post_selection_filter_fn(
        qubit_count, n_electrons + 1, sz=0.5
    )
    assert filter_fn(0b11001)  # [0, 1, 2]
    assert not filter_fn(0b1011)  # [0, 1, 3]
    assert filter_fn(22)  # [2, 3, 4]
    assert filter_fn(0b110)  # [2, 4, 7]
    assert not filter_fn(48)  # [3, 5, 6]

    filter_fn = create_scbk_electron_number_post_selection_filter_fn(
        qubit_count, 2, sz=1.0
    )
    assert filter_fn(0b1)  # [0, 2]
    assert filter_fn(0b111)  # [0, 4]
    assert filter_fn(0b11)  # [0, 6]
    assert filter_fn(0b110)  # [2, 4]
    assert filter_fn(0b10)  # [2, 6]
    assert filter_fn(0b100)  # [4, 6]
    assert not filter_fn(0b11011)  # [0, 1]
    assert not filter_fn(0b110000)  # [3, 5]
    assert not filter_fn(0b10010)  # [2, 3]

    filter_fn = create_scbk_electron_number_post_selection_filter_fn(
        qubit_count, 3, sz=-0.5
    )
    assert filter_fn(0b1011)  # [0, 1, 3]
    assert filter_fn(0b111011)  # [0, 1, 5]
    assert filter_fn(0b11011)  # [0, 1, 7]
    assert filter_fn(0b110011)  # [0, 3, 5]
    assert filter_fn(0b10011)  # [0, 3, 7]
    assert filter_fn(0b100011)  # [0, 5, 7]
    assert filter_fn(0b1010)  # [1, 2, 3]
    assert filter_fn(0b111010)  # [1, 2, 5]
    assert filter_fn(0b11010)  # [1, 2, 7]
    assert filter_fn(0b1100)  # [1, 3, 4]
    assert filter_fn(0b1000)  # [1, 3, 6]
    assert filter_fn(0b1100)  # [1, 4, 5]
    assert not filter_fn(0b11001)  # [0, 1, 2]
    assert not filter_fn(0b11111)  # [0, 1, 4]
    assert not filter_fn(0b1)  # [0, 2, 7]
    assert not filter_fn(0b111)  # [0, 4, 6]
    assert not filter_fn(0b11110)  # [1, 2, 4]
