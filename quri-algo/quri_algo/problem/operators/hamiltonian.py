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
    def get_matrix_representation(self, *args: Any) -> npt.NDArray[np.complex128]:
        ...


HamiltonianT = TypeVar("HamiltonianT", bound="Hamiltonian")


class QubitHamiltonian(Hamiltonian):
    """Represents the Hamiltonian of a fixed qubit size in its qubit
    Hamiltonian form."""

    def __init__(self, n_qubit: int, qubit_hamiltonian: Operator):
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
