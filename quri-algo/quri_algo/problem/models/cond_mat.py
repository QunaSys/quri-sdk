from dataclasses import dataclass

from openfermion import fermi_hubbard
from quri_parts.openfermion.transforms import OpenFermionQubitMapping

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
            self.get_fermionic_hamiltonian().fermionic_hamiltonian
        )
        return QubitHamiltonian(mapping.n_qubits, qubit_operator)

    def get_fermionic_hamiltonian(self) -> FermionicHamiltonian:
        return FermionicHamiltonian(
            2 * self.nx * self.ny,
            fermi_hubbard(
                self.nx, self.ny, self.t, self.U, self.mu, periodic=self.periodic
            ),
        )
