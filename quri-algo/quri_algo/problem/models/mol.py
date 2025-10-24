# Licensed under the MIT License (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence, cast
from typing_extensions import TypeAlias
from functools import cached_property

from pyscf import gto, scf, df
from openfermion.ops import FermionOperator
from openfermion.ops.representations.interaction_operator import InteractionOperator


from quri_parts.core.operator.operator import Operator
from quri_parts.pyscf.mol import get_spin_mo_integrals_from_mole
from quri_parts.openfermion.mol import (
    get_fermionic_hamiltonian as get_fermionic_mapped_hamiltonian,
    operator_from_of_fermionic_op,
)
from quri_parts.openfermion.transforms import jordan_wigner
from quri_parts.chem.mol import ActiveSpace, cas

from quri_algo.problem.operators.hamiltonian import (
    FermionicHamiltonian,
    QubitHamiltonian,
)
from quri_algo.problem.models.interface import HamiltonianMixin


AtomCoordinate: TypeAlias = tuple[float, float, float]


@dataclass
class MolecularSystem(HamiltonianMixin):
    """Represents a molecular system using PySCF as a backend.

    Provides utilities for:
      - Building the molecule
      - Performing a Hartree–Fock (RHF or ROHF) calculation
      - Mapping the resulting fermionic Hamiltonian into a qubit Hamiltonian
        compatible with QURI's quantum chemistry interface.
    """

    atom: Sequence[tuple[str, AtomCoordinate]] | str
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    frozen: Optional[list[int]] = None
    backend: Literal["pyscf_mem_efficient", "pyscf_density_fitting"] = (
        "pyscf_mem_efficient"
    )

    @cached_property
    def pyscf_mol(self) -> gto.Mole:
        """PySCF Mole object (built once and cached)."""
        return gto.M(
            atom=self.atom,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            unit="Angstrom",
            symmetry=False,
        )

    def get_pyscf_molecule(self) -> gto.Mole:
        return self.pyscf_mol

    @cached_property
    def hartree_fock(self) -> scf.hf.SCF:
        """Run RHF/ROHF once and cache result."""
        pyscf_scf = (
            scf.ROHF(self.pyscf_mol) if self.pyscf_mol.spin else scf.RHF(self.pyscf_mol)
        )
        if self.backend == "pyscf_mem_efficient":
            pyscf_scf.direct_scf = True
        elif self.backend == "pyscf_density_fitting":
            pyscf_scf = pyscf_scf.density_fit()
            pyscf_scf.with_df._cderi = df.incore.cholesky_eri(self.pyscf_mol)
        pyscf_scf.run(verbose=0)
        if not pyscf_scf.converged:
            raise RuntimeError("PySCF Hartree–Fock failed to converge.")
        return pyscf_scf

    def get_hartree_fock(self) -> scf.hf.SCF:
        return self.hartree_fock

    @cached_property
    def active_space(self) -> ActiveSpace:
        pyscf_mol = self.pyscf_mol
        n_frozen = len(self.frozen) if self.frozen else 0
        n_orbitals = pyscf_mol.nao - n_frozen
        n_electrons = pyscf_mol.nelectron - 2 * n_frozen
        active_orbs_indices = None
        if self.frozen:
            active_orbs_indices = [
                i for i in range(pyscf_mol.nao) if i not in self.frozen
            ]
        return cas(
            n_active_ele=n_electrons,
            n_active_orb=n_orbitals,
            active_orbs_indices=active_orbs_indices,
        )

    def get_active_space(self) -> ActiveSpace:
        return self.active_space

    @cached_property
    def fermionic_hamiltonian(self) -> FermionicHamiltonian:
        as_eint_set, mo_eint_set = get_spin_mo_integrals_from_mole(
            self.pyscf_mol, self.hartree_fock.mo_coeff, self.active_space
        )
        fermion_op = get_fermionic_mapped_hamiltonian(mo_eint_set)
        n_spin_orbital = self.hartree_fock.mo_coeff.shape[1] * 2
        return FermionicHamiltonian(
            n_spin_orbital=n_spin_orbital,
            fermionic_hamiltonian=cast(FermionOperator, fermion_op),
        )

    def get_fermionic_hamiltonian(self) -> FermionicHamiltonian:
        return self.fermionic_hamiltonian

    @cached_property
    def qubit_hamiltonian(self) -> QubitHamiltonian:
        """Return the Jordan–Wigner–mapped qubit Hamiltonian."""
        fermion_h = self.fermionic_hamiltonian
        n_qubit = self.active_space.n_active_orb * 2

        qubit_operator, _ = operator_from_of_fermionic_op(
            fermion_h.fermionic_hamiltonian,
            self.active_space,
            sz=None,
            fermion_qubit_mapping=jordan_wigner,
        )

        return QubitHamiltonian(n_qubit=n_qubit, qubit_hamiltonian=qubit_operator)

    def get_qubit_hamiltonian(self) -> QubitHamiltonian:
        return self.qubit_hamiltonian
