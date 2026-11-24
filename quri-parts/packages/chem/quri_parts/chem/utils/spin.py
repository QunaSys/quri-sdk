# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Collection, Optional


def occupation_state_sz(occ_indices: Collection[int]) -> float:
    n_up = 0
    n_down = 0
    for i in occ_indices:
        if i % 2:
            n_down += 1
        else:
            n_up += 1
    return (n_up - n_down) / 2


def reference_occupation(
    n_spin_orbitals: int, n_fermions: int, sz: Optional[float] = None
) -> list[int]:
    """Returns the occupied spin orbitals of the reference state for a spin
    sector.

    Even indices are spin up and odd indices are spin down. The reference fills
    the lowest ``n_fermions / 2 + sz`` spin-up orbitals and the lowest
    ``n_fermions / 2 - sz`` spin-down orbitals. This is the reference that
    :class:`~quri_parts.openfermion.ansatz.TrotterUCCSD` and
    :class:`~quri_parts.chem.ansatz.AllSinglesDoubles` build their excitations
    from.

    Args:
        n_spin_orbitals: Number of spin orbitals.
        n_fermions: Number of fermions.
        sz: Spin projection of the reference state. If ``None``, the lowest
            :attr:`n_fermions` spin orbitals are occupied, which is the most
            balanced sector.

    Raises:
        ValueError: If :attr:`n_fermions` is not in ``[0, n_spin_orbitals]``, or
            if no state with :attr:`n_fermions` fermions in
            :attr:`n_spin_orbitals` spin orbitals has this ``sz``.
    """
    if not 0 <= n_fermions <= n_spin_orbitals:
        raise ValueError(
            f"n_fermions={n_fermions} is not possible for "
            f"{n_spin_orbitals} spin orbitals."
        )
    if sz is None:
        return list(range(n_fermions))
    twice_sz = 2.0 * sz
    n_up = (n_fermions + int(twice_sz)) // 2
    n_down = n_fermions - n_up
    if not (
        twice_sz.is_integer()
        and (n_fermions + int(twice_sz)) % 2 == 0
        and 0 <= n_up <= (n_spin_orbitals + 1) // 2
        and 0 <= n_down <= n_spin_orbitals // 2
    ):
        raise ValueError(
            f"Invalid sz={sz} for {n_fermions} fermions in "
            f"{n_spin_orbitals} spin orbitals."
        )
    up = [2 * i for i in range(n_up)]
    down = [2 * i + 1 for i in range(n_down)]
    return sorted(up + down)
