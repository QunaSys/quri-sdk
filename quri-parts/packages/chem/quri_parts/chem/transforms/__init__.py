# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Collection, Sequence
from typing import Callable, Optional, Protocol

from typing_extensions import TypeAlias

from quri_parts.core.state import ComputationalBasisState

#: Interface for a function that maps a collection of occupied spin orbital indices to
#: a computational basis state of qubits.
#: Note that the mapping does not depend on the order of the occupied indices.
#: A computational basis state with a positive sign should always be returned.
FermionQubitStateMapper: TypeAlias = Callable[
    [Collection[int]], ComputationalBasisState
]

#: Interface for a function that maps a computational basis state of qubits to
#: a collection of occupied spin orbital indices.
QubitFermionStateMapper: TypeAlias = Callable[
    [ComputationalBasisState], Collection[int]
]


class FermionCreationTerm:
    """Represents a term for creating a single Fermionic Fock state with a
    coefficient.

    ``indices`` specify indices of occupied spin orbitals. The orbitals
    should be indexed in an alternating order of up and down spins
    (especially important when using symmetry-conserving Bravyi-Kitaev
    mapping). The order of indices in ``indices`` corresponds to the
    order of Fermionic creation operators. Therefore this class also
    contains information about a sign coming from ordering of creation
    operators.
    """

    def __init__(self, indices: Sequence[int], coef: complex = 1.0):
        self.coef = coef * (-1) ** self._inversion_number(indices)
        self.indices = sorted(indices)

    @staticmethod
    def _inversion_number(arr: Sequence[int]) -> int:
        length = len(arr)
        inv = 0
        for i in range(length):
            inv += sum(arr[i] > arr[k] for k in range(i + 1, length))
        return inv


class FermionQubitMapperFactory(Protocol):
    """Mapping from Fermionic states to qubit states."""

    @staticmethod
    @abstractmethod
    def n_qubits_required(n_spin_orbitals: int) -> int:
        """Returns a number of qubits the mapping requires for a given number
        of spin orbitals."""
        ...

    @staticmethod
    @abstractmethod
    def n_spin_orbitals(n_qubits: int) -> int:
        """Returns a number of spin orbitals that the mapping can represent
        with a given number of qubits."""
        ...

    @abstractmethod
    def get_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> FermionQubitStateMapper:
        """Returns a function that maps occupied spin orbital indices to a
        computational basis state of qubits.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits.
            n_fermions:
                When specified, restrict the mapping to a subspace spanned by states
                containing the fixed number of Fermions. Some mappings require this
                argument (e.g. symmetry-conserving Bravyi-Kitaev transformation) while
                the others ignore it.
            sz:
                The spin along the z-axis of the state you want to transform. Some
                mappings require this argument (e.g. symmetry-conserving Bravyi-Kitaev
                transformation) while the others ignore it.
        """
        ...

    @abstractmethod
    def get_inv_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> QubitFermionStateMapper:
        """Returns a function that maps a computational basis state of qubits
        to the set of occupied spin orbital indices.

        Args:
            n_spin_orbitals:
                The number of spin orbitals.
            n_fermions:
                The number of fermions considered when the qubit state is mapped.
                Some mappings require this argument (e.g. symmetry-conserving
                Bravyi-Kitaev transformation) while the others ignore it.
            sz:
                The spin along the z-axis of the state you want to transform. Some
                mappings require this argument (e.g. symmetry-conserving Bravyi-Kitaev
                transformation) while the others ignore it.
        """
        ...


class FermionQubitMapping(Protocol):
    """Mapping from Fermionic states to qubit states."""

    @abstractproperty
    def n_spin_orbitals(self) -> int:
        """Number of spin orbitals this mapping object works with."""
        ...

    @abstractproperty
    def n_qubits(self) -> int:
        """Number of qubits this mapping object works with."""
        ...

    @abstractproperty
    def state_mapper(self) -> "FermionQubitStateMapper":
        """Returns a function that maps occupied spin orbital indices to a
        computational basis state of qubits."""
        ...

    @abstractproperty
    def inv_state_mapper(self) -> "QubitFermionStateMapper":
        """Returns a function that maps a computational basis state of qubits
        to the set of occupied spin orbital indices."""
        ...


def jordan_wigner_n_qubits_required(n_spin_orbitals: int) -> int:
    """Number of qubits required to perform Jordan-Wigner mapping for a system
    with `n_spin_orbtals` spin orbitals."""
    return n_spin_orbitals


def jordan_wigner_n_spin_orbitals(n_qubits: int) -> int:
    """Returns a number of spin orbitals that the Jordan-Wigner mapping can
    represent with a given number of qubits."""
    return n_qubits


class JordanWignerMapperFactory(FermionQubitMapperFactory, ABC):
    """Jordan-Wigner transformation."""

    @staticmethod
    def n_qubits_required(n_spin_orbitals: int) -> int:
        """Number of qubits required to perform Jordan-Wigner mapping for a
        system with `n_spin_orbtals` spin orbitals."""
        return jordan_wigner_n_qubits_required(n_spin_orbitals)

    @staticmethod
    def n_spin_orbitals(n_qubits: int) -> int:
        """Returns the number of spin orbitals that the Jordan-Wigner mapping
        can represent with a given number of qubits."""
        return jordan_wigner_n_spin_orbitals(n_qubits)


class JordanWigner(FermionQubitMapping, ABC):
    @property
    def n_qubits(self) -> int:
        """Number of qubits to perform Jordan-Wigner mapping with."""
        return jordan_wigner_n_qubits_required(self.n_spin_orbitals)


def bravyi_kitaev_n_qubits_required(n_spin_orbtals: int) -> int:
    """Number of qubits required to perform Bravyi-Kitaev mapping for a system
    with `n_spin_orbtals` spin orbitals."""
    return n_spin_orbtals


def bravyi_kitaev_n_spin_orbitals(n_qubits: int) -> int:
    """Returns the number of spin orbitals that the Bravyi-Kitaev mapping can
    represent with a given number of qubits."""
    return n_qubits


class BravyiKitaevMapperFactory(FermionQubitMapperFactory, ABC):
    """Bravyi-Kitaev transformation."""

    @staticmethod
    def n_qubits_required(n_spin_orbitals: int) -> int:
        """Number of qubits required to perform Bravyi-Kitaev mapping for a
        system with `n_spin_orbtals` spin orbitals."""
        return bravyi_kitaev_n_qubits_required(n_spin_orbitals)

    @staticmethod
    def n_spin_orbitals(n_qubits: int) -> int:
        """Returns the number of spin orbitals that the Bravyi-Kitaev mapping
        can represent with a given number of qubits."""
        return bravyi_kitaev_n_spin_orbitals(n_qubits)


class BravyiKitaev(FermionQubitMapping, ABC):
    @property
    def n_qubits(self) -> int:
        """Number of qubits to perform Bravyi-Kitaev mapping with."""
        return bravyi_kitaev_n_qubits_required(self.n_spin_orbitals)


def symmetry_conserving_bravyi_kitaev_n_qubits_required(n_spin_orbtals: int) -> int:
    """Number of qubits required to perform symmetry-conserving Bravyi-Kitaev
    mapping for a system with `n_spin_orbtals` spin orbitals."""
    return n_spin_orbtals - 2


def symmetry_conserving_bravyi_kitaev_n_spin_orbitals(n_qubits: int) -> int:
    """Returns the number of spin orbitals that the symmetry-conservinh Bravyi-
    Kitaev mapping can represent with a given number of qubits."""
    return n_qubits + 2


class SymmetryConservingBravyiKitaevMapperFactory(FermionQubitMapperFactory, ABC):
    """Symmetry-conserving Bravyi-Kitaev transformation described in
    arXiv:1701.08213.

    Note that in this mapping the spin orbital indices are first
    reordered to all spin-up orbitals, then all spin-down orbitals.
    Bravyi-Kitaev transoformation is applied after the reordering and
    then two qubits are dropped using conservation of particle number
    and spin.
    """

    @staticmethod
    def n_qubits_required(n_spin_orbitals: int) -> int:
        """Number of qubits required to perform symmetry-conserving Bravyi-
        Kitaev mapping for a system with `n_spin_orbtals` spin orbitals."""
        return symmetry_conserving_bravyi_kitaev_n_qubits_required(n_spin_orbitals)

    @staticmethod
    def n_spin_orbitals(n_qubits: int) -> int:
        """Returns the number of spin orbitals that the symmetry-conserving
        Bravyi-Kitaev mapping can represent with a given number of qubits."""
        return symmetry_conserving_bravyi_kitaev_n_spin_orbitals(n_qubits)


class SymmetryConservingBravyiKitaev(FermionQubitMapping, ABC):
    @property
    def n_qubits(self) -> int:
        """Number of qubits to perform symmetry-conserving Bravyi-Kitaev
        mapping with."""
        return symmetry_conserving_bravyi_kitaev_n_qubits_required(self.n_spin_orbitals)
