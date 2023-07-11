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

from quri_parts.chem.utils.post_selection_filters import (
    _inv_bk_trans_mat,
    create_bk_electron_number_post_selection_filter_fn,
    create_jw_electron_number_post_selection_filter_fn,
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


def test_inv_bk_trans_mat() -> None:
    assert np.array_equal(_inv_bk_trans_mat(0), np.array([1]))
    assert np.array_equal(_inv_bk_trans_mat(1), np.array([[1, 0], [1, 1]]))
    assert np.array_equal(
        _inv_bk_trans_mat(2),
        np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 1]]),
    )
    assert np.array_equal(
        _inv_bk_trans_mat(3),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1, 1, 1],
            ]
        ),
    )
    assert np.array_equal(
        _inv_bk_trans_mat(4),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            ]
        ),
    )
