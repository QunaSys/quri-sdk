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

from quri_parts.algo.mitigation.post_selection import PostSelectionFilterFunction
from quri_parts.chem.utils.spin import occupation_state_sz
from quri_parts.core.state import ComputationalBasisState
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    symmetry_conserving_bravyi_kitaev,
)


def create_jw_electron_number_post_selection_filter_fn(
    n_electrons: int, sz: Optional[float] = None
) -> PostSelectionFilterFunction:
    """Returns :class:`PostSelectionFilterFunction` that checks if the number
    of occupied orbitals matchs given ``n_electrons`` (and ``sz`` if
    specified).

    Here ``bits`` is a bitstring obtained by measuring the states mapped
    by Jordan-Wigner transformation.
    """

    def filter_fn(bits: int) -> bool:
        if sz is not None:
            n_up = bin(bits)[-1:1:-2].count("1")
            n_down = bin(bits)[-2:1:-2].count("1")
            if n_up - n_down != sz * 2:
                return False
        return bin(bits).count("1") == n_electrons

    return filter_fn


def create_bk_electron_number_post_selection_filter_fn(
    qubit_count: int, n_electrons: int, sz: Optional[float] = None
) -> PostSelectionFilterFunction:
    """Returns :class:`PostSelectionFilterFunction` that checks if the number
    of occupied orbitals matchs given ``n_electrons`` (and ``sz`` if
    specified).

    Here ``bits`` is a bitstring obtained by measuring the states mapped
    by Bravyi-Kitaev transformation.
    """
    inv_st_mapper = bravyi_kitaev.get_inv_state_mapper(
        bravyi_kitaev.n_spin_orbitals(qubit_count)
    )

    def filter_fn(bits: int) -> bool:
        state = ComputationalBasisState(qubit_count, bits=bits)
        occ_indices = inv_st_mapper(state)
        if sz is not None and occupation_state_sz(occ_indices) != sz:
            return False
        return len(occ_indices) == n_electrons

    return filter_fn


def create_scbk_electron_number_post_selection_filter_fn(
    qubit_count: int, n_electrons: int, sz: float
) -> PostSelectionFilterFunction:
    """Returns :class:`PostSelectionFilterFunction` that checks if the number
    of occupied orbitals matchs given ``n_electrons``.

    Here ``bits`` is a bitstring obtained by measuring the states mapped
    by symmetry-conserving Bravyi-Kitaev transformation.
    """
    inv_st_mapper = symmetry_conserving_bravyi_kitaev.get_inv_state_mapper(
        symmetry_conserving_bravyi_kitaev.n_spin_orbitals(qubit_count),
        n_electrons,
        sz,
    )

    def filter_fn(bits: int) -> bool:
        state = ComputationalBasisState(qubit_count, bits=bits)
        occ_indices = inv_st_mapper(state)

        if occupation_state_sz(occ_indices) != sz:
            return False
        return len(occ_indices) == n_electrons

    return filter_fn
