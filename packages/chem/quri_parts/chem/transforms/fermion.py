# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Optional


class FermionLabel(tuple[tuple[int, int]]):
    """FermionLabel represents a label for a Fermion string, e.g. ``0^ 1
    2^``."""

    """tuple[index, is_dag(1 means dagger)], reference PauliLabel"""

    def __new__(cls, arg: Sequence[tuple[int, int]] = ()) -> "FermionLabel":
        instance = super().__new__(cls, tuple(arg))  # type: ignore
        return instance

    def __str__(self) -> str:
        return " ".join(
            f"{index}{('^' if bool(is_dag) else '')}" for index, is_dag in self
        )

    def fermion_indices(self) -> list[int]:
        """Returns a list of fermion indices on which fermion operator act.

        Note that this list does not have a definite ordering.
        """
        return list(map(lambda t: t[0], self))

    def fermion_at(self, index: int) -> Optional[int]:
        """Returns a fermion at the fermion index ``index``.

        Returns ``None`` if no fermion operator acts on the index.
        """
        for i, o in self:
            if i == index:
                return o
        return None

    @property
    def index_and_fermion_id_list(self) -> tuple[Sequence[int], Sequence[int]]:
        """A pair of a list of indices of fermions on which the fermion
        operator act, and a list of the fermion operators in the order
        corresponding to the index list."""
        index_list, pauli_list = zip(*self)
        return list(index_list), list(pauli_list)

    @staticmethod
    def from_index_and_fermion_list(
        index: Sequence[int], pauli: Sequence[int]
    ) -> "FermionLabel":
        """Create a PauliLabel from a list of qubit indices and a list of Pauli
        matrices."""
        if len(index) != len(pauli):
            raise ValueError("Length of index and fermion unmatch")
        return FermionLabel(tuple(zip(index, pauli)))


def fermion_label(p: Sequence[tuple[int, int]]) -> FermionLabel:
    """Create a FermionLabel.

    The argument can be one of the followings:

    * An Iterable of tuple[int, int], pairs of a fermion index.
    """
    if isinstance(p, Sequence):
        return FermionLabel(p)
    else:
        raise ValueError(f"Invalid argument for pauli_label: {p}")
