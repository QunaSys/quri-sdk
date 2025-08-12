# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence


def get_core_and_active_orbital_indices(
    n_active_ele: int,
    n_active_orb: int,
    n_electrons: int,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> tuple[Sequence[int], Sequence[int]]:
    """Returns sequences of spatial occupied orbital indices and active
    orbitals indices obtained from the number of active electrons, number of
    active orbitals, and number of total electrons.

    Args:
        n_active_ele:
            number of active electrons.
        n_active_orb:
            number of active orbitals.
        n_electrons:
            total number of electrons.
        active_orbs_indices:
            sequence of spatial orbital indices of the active orbitals.
    """
    n_electrons_core = n_electrons - n_active_ele
    if n_electrons_core % 2 == 1:
        raise ValueError("The number of electrons in core must be even.")
    core_orb = n_electrons_core // 2
    if not active_orbs_indices:
        occupied_indices = list(range(core_orb))
        active_indices = [core_orb + i for i in range(n_active_orb)]

    else:
        if len(active_orbs_indices) != n_active_orb:
            raise ValueError(
                "The size of active_orbs_indices must be consistent with"
                " n_active_orb."
            )
        active_indices = list(active_orbs_indices)
        n_orb = core_orb + n_active_orb
        occupied_indices = []
        for i in range(n_orb):
            if i not in active_indices:
                occupied_indices.append(i)
            if len(occupied_indices) == core_orb:
                break

    return occupied_indices, active_indices


def convert_to_spin_orbital_indices(
    occupied_indices: Sequence[int], active_indices: Sequence[int]
) -> tuple[Sequence[int], Sequence[int]]:
    """Convert each spatial occupied orbital indices and active orbitals
    indices into the spin occupied orbital indices and active spin indices.

    Args:
        occupied_indices:
            A Sequence of spatial occupied orbital indices.
        active_indices:
            A Sequence of spatial active orbital indices.
    """
    occupied_spin_indices = []
    active_spin_indices = []

    for o_i in occupied_indices:
        occupied_spin_indices.extend([2 * o_i, 2 * o_i + 1])
    for a_i in active_indices:
        active_spin_indices.extend([2 * a_i, 2 * a_i + 1])
    return occupied_spin_indices, active_spin_indices
