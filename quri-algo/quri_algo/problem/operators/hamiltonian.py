# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from openfermion import FermionOperator
from quri_parts.core.operator import Operator


class Hamiltonian(ABC):
    """Represents an encoded Hamiltonian."""

    @abstractmethod
    def get_matrix_representation(self, *args: Any) -> npt.NDArray[np.complex128]: ...


HamiltonianT = TypeVar("HamiltonianT", bound="Hamiltonian")


class QubitHamiltonian(Hamiltonian):
    """A Hamiltonian expressed in qubit (Pauli operator) form for a fixed-size qubit system.

    This class represents a molecular or physical system's Hamiltonian that has been
    mapped to a qubit representation. The resulting operator acts on a
    Hilbert space of dimension :math:`2^{n_qubit}` and is expressed as a linear
    combination of Pauli strings with real or complex coefficients.

    The explicit qubit count is tracked to ensure consistency between the Hamiltonian
    and quantum circuits or states.

    Attributes:
        n_qubit (int): The total number of qubits (spin orbitals) the Hamiltonian acts on.
        _qubit_hamiltonian (Operator): The underlying qubit-space operator, typically a
            sum of weighted Pauli strings.

    """

    def __init__(self, n_qubit: int, qubit_hamiltonian: Operator):
        """Initialise an instance of :class:`.QubitHamiltonian`.

        Args:
            n_qubit (int): Number of qubits (i.e., spin orbitals after any active-space
                truncation). This defines the size of the Hilbert space on which the
                Hamiltonian acts.
            qubit_hamiltonian (Operator): A QURI Parts style Operator.
        """
        self.n_qubit = n_qubit
        self._qubit_hamiltonian = qubit_hamiltonian

    @property
    def qubit_hamiltonian(self) -> Operator:
        return self._qubit_hamiltonian.copy()

    def get_matrix_representation(
        self, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.complex128]:
        raise NotImplementedError("Not supported yet")


class FermionicHamiltonian(Hamiltonian):
    def __init__(self, n_spin_orbital: int, fermionic_hamiltonian: FermionOperator):
        self.n_spin_orbital = n_spin_orbital
        self._fermionic_hamiltonian = fermionic_hamiltonian

    @property
    def fermionic_hamiltonian(self) -> FermionOperator:
        return self._fermionic_hamiltonian

    def get_matrix_representation(
        self, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.complex128]:
        raise NotImplementedError("Not supported yet")
