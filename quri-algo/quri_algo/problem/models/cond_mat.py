# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quri_parts.openfermion.transforms import OpenFermionQubitMapping
try:
    from openfermion import fermi_hubbard
except ImportError:
    raise ImportError("openfermion is not installed")

from quri_algo.problem.operators.hamiltonian import (
    FermionicHamiltonian,
    QubitHamiltonian,
)

from .interface import HamiltonianMixin


@dataclass
class FermiHubbardSystem(HamiltonianMixin):
    nx: int
    ny: int
    t: float
    U: float
    mu: float
    periodic: bool

    def get_qubit_hamiltonian(
        self, mapping: OpenFermionQubitMapping
    ) -> QubitHamiltonian:
        qubit_operator = mapping.of_operator_mapper(
            self.get_fermionic_hamiltonian().fermion_operator
        )
        return QubitHamiltonian(mapping.n_qubits, qubit_operator)

    def get_fermionic_hamiltonian(self) -> FermionicHamiltonian:
        return FermionicHamiltonian(
            2 * self.nx * self.ny,
            fermi_hubbard(
                self.nx, self.ny, self.t, self.U, self.mu, periodic=self.periodic
            ),
        )
