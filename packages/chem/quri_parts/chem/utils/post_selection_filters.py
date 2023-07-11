# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import numpy.typing as npt

from quri_parts.algo.mitigation.post_selection import PostSelectionFilterFunction


def create_jw_electron_number_post_selection_filter_fn(
    n_electrons: int, sz: Optional[float] = None
) -> PostSelectionFilterFunction:
    """Returns `class`:PostSelectionFilterFunction: that checks if the number
    of occupied orbitals matchs given ``n_electrons`` (and ``sz`` if
    specified).

    Here ``bits`` is a bitstring obtained by measuring the states mapped
    by Jordan-Wigner transformation.
    """

    def filter_fn(bits: int) -> bool:
        if sz:
            n_up = bin(bits)[-1:1:-2].count("1")
            n_down = bin(bits)[-2:1:-2].count("1")
            if n_up - n_down != sz * 2:
                return False
        return bin(bits).count("1") == n_electrons

    return filter_fn


def create_bk_electron_number_post_selection_filter_fn(
    qubit_count: int, n_electrons: int, sz: Optional[float] = None
) -> PostSelectionFilterFunction:
    """Returns `class`:PostSelectionFilterFunction: that checks if the number
    of occupied orbitals matchs given ``n_electrons`` (and ``sz`` if
    specified).

    Here ``bits`` is a bitstring obtained by measuring the states mapped
    by Bravyi-Kitaev transformation.
    """
    pow_base_2 = np.ceil(np.log2(qubit_count)).astype(np.uint8)
    inv_trans_mat = _inv_bk_trans_mat(pow_base_2)[:qubit_count, :qubit_count]

    def filter_fn(bits: int) -> bool:
        bk_state_array = (bits >> np.arange(qubit_count) & 1).astype(np.uint8)
        occ_num_state = np.dot(inv_trans_mat, bk_state_array) % 2
        if sz:
            n_up = np.count_nonzero(occ_num_state[::2] == 1)
            n_down = np.count_nonzero(occ_num_state[1::2] == 1)
            if n_up - n_down != sz * 2:
                return False
        return np.count_nonzero(occ_num_state == 1) == n_electrons

    return filter_fn


def _inv_bk_trans_mat(log_2_dim: int) -> npt.NDArray[np.uint8]:
    """Function that generate inverse Bravyi-Kitaev transformation matrix
    recursively.

    So the function returns the matrix which converts BK mapped state to
    occupation number state.
    """
    if log_2_dim == 0:
        return np.array([1])
    mat = np.zeros((2**log_2_dim, 2**log_2_dim), dtype=np.uint8)
    partial = _inv_bk_trans_mat(log_2_dim - 1)
    mat[0 : 2 ** (log_2_dim - 1), 0 : 2 ** (log_2_dim - 1)] = partial  # noqa
    mat[2 ** (log_2_dim - 1) :, 2 ** (log_2_dim - 1) :] = partial  # noqa
    mat[-1, 2 ** (log_2_dim - 1) - 1] = 1
    return mat
