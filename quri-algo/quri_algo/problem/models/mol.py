# Licensed under the MIT License (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Optional, Sequence, Union, cast

try:
    from openfermion.ops import FermionOperator
except ImportError:
    ImportError("openfermion is not installed")
try:
    from pyscf import df, gto, scf
except ImportError:
    raise ImportError("pyscf is not installed")

from quri_parts.chem.mol import ActiveSpace, cas
from quri_parts.core.operator import Operator
from quri_parts.core.state import ComputationalBasisState
from quri_parts.openfermion.mol import (
    get_fermionic_hamiltonian as get_fermionic_mapped_hamiltonian,
)
from quri_parts.openfermion.mol import operator_from_of_fermionic_op
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapperFactory,
    OpenFermionQubitMapping,
    jordan_wigner,
)
from quri_parts.pyscf.mol import get_spin_mo_integrals_from_mole
from typing_extensions import TypeAlias

from quri_algo.problem.models.interface import HamiltonianMixin
from quri_algo.problem.operators.hamiltonian import (
    FermionicHamiltonian,
    QubitHamiltonian,
)

AtomCoordinate: TypeAlias = tuple[float, float, float]


@dataclass
class MolecularSystem(HamiltonianMixin):
    """Represents a molecular system using PySCF as a backend.

    Provides utilities for:
      - Building the molecule, or wrapping an already-built one
        (:meth:`from_pyscf`)
      - Performing a Hartree–Fock (RHF or ROHF) calculation, or reusing an
        already-converged one (:meth:`from_pyscf`)
      - Mapping the resulting fermionic Hamiltonian into a qubit Hamiltonian
        compatible with QURI's quantum chemistry interface, for any
        supported fermion-to-qubit mapping
      - Deriving a Hartree-Fock reference as a computational basis state,
        a natural seed for subspace/adaptive algorithms such as QSCI

    All PySCF and mapping work is performed lazily and cached on first
    access, whether the instance was built via the default constructor or
    via :meth:`from_pyscf`: constructing an instance does not by itself run
    an SCF calculation.

    Attributes:
        atom (Sequence[tuple[str, tuple[float, float, float]]] | str):
            The list of atoms with their coordinates (in Angstroms),
            e.g., [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))],
            or a PySCF-readable string.
        basis (str): Basis set to use for the molecule (default: "sto-3g").
        charge (int): Molecular charge (default: 0).
        spin (int): Total spin (2S) of the molecule (default: 0).
        frozen (Optional[list[int]]): Indices of frozen orbitals to exclude from the active space.
            Ignored (superseded) when using :meth:`from_pyscf` with an explicit ``active_space``.
        backend (Literal["pyscf_mem_efficient", "pyscf_density_fitting"]):
            SCF computation backend used when this instance builds and runs its own
            RHF/ROHF; either memory-efficient or density-fitting (default: "pyscf_mem_efficient").
            Has no effect when :meth:`from_pyscf` is given an already-converged mean field.
        fermion_qubit_mapping (OpenFermionQubitMapperFactory): Fermion-to-qubit mapping
            factory used for :attr:`qubit_hamiltonian` and :attr:`hf_state` (default:
            Jordan-Wigner).
        sz (Optional[float]): Target Sz passed through to the fermion-qubit mapping.
    """

    atom: Sequence[tuple[str, AtomCoordinate]] | str
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    frozen: Optional[list[int]] = None
    backend: Literal[
        "pyscf_mem_efficient", "pyscf_density_fitting"
    ] = "pyscf_mem_efficient"
    fermion_qubit_mapping: OpenFermionQubitMapperFactory = jordan_wigner
    sz: Optional[float] = None

    def __post_init__(self) -> None:
        self._active_space_override: Optional[ActiveSpace] = None

    @classmethod
    def from_pyscf(
        cls,
        mol_or_mf: Union["gto.Mole", "scf.hf.SCF"],
        active_space: Optional[ActiveSpace] = None,
        *,
        sz: Optional[float] = None,
        fermion_qubit_mapping: OpenFermionQubitMapperFactory = jordan_wigner,
    ) -> "MolecularSystem":
        """Build a :class:`MolecularSystem` from an existing PySCF object.

        Unlike the default constructor (which builds its own
        :class:`~pyscf.gto.Mole` from ``atom``/``basis``/``charge``/``spin``
        and always runs a fresh RHF/ROHF), this wraps whatever the caller
        already has:

        - Given a bare :class:`~pyscf.gto.Mole`, Hartree-Fock is still run
          lazily on first access (using the default RHF/ROHF + ``backend``
          logic, with a ``newton()`` fallback on non-convergence), exactly
          like the default constructor. Passing one in does not by itself
          force a calculation.
        - Given an already-converged mean field, it is used as-is and never
          rerun, preserving whatever method, SCF settings, or convergence
          handling (e.g. a custom ``conv_tol``, density fitting with a
          specific auxiliary basis, a ``chkfile`` restart) the caller
          already configured.

        Args:
            mol_or_mf: A :class:`~pyscf.gto.Mole`, or an already-converged
                restricted (RHF/ROHF) PySCF mean field. Unrestricted (UHF)
                mean fields are not supported.
            active_space: Active space to restrict to. ``None`` derives the
                full space from the molecule (equivalent to ``frozen=None``
                on the default constructor).
            sz: Target Sz passed through to the fermion-qubit mapping.
            fermion_qubit_mapping: Fermion-to-qubit mapping factory (default
                Jordan-Wigner).

        Returns:
            MolecularSystem: A new instance wrapping ``mol_or_mf``.

        Raises:
            NotImplementedError: If ``mol_or_mf`` is an unrestricted (UHF)
                mean field.
            RuntimeError: If ``mol_or_mf`` is a mean field that has not
                converged.
        """
        mf: Optional[scf.hf.SCF]
        if isinstance(mol_or_mf, gto.Mole):
            mol, mf = mol_or_mf, None
        else:
            mf = mol_or_mf
            if isinstance(mf, scf.uhf.UHF):
                raise NotImplementedError(
                    "MolecularSystem.from_pyscf does not support unrestricted "
                    "(UHF) mean fields; only RHF/ROHF-shaped mo_coeff is "
                    "supported."
                )
            if not getattr(mf, "converged", True):
                raise RuntimeError("Provided PySCF mean field is not converged.")
            mol = mf.mol

        instance = cls.__new__(cls)
        instance.atom = mol.atom
        instance.basis = mol.basis
        instance.charge = mol.charge
        instance.spin = mol.spin
        instance.frozen = None
        instance.backend = "pyscf_mem_efficient"
        instance.fermion_qubit_mapping = fermion_qubit_mapping
        instance.sz = sz
        instance._active_space_override = active_space
        # Pre-seed the cached_property caches below: assigning directly to a
        # cached_property-backed attribute stores the value in __dict__,
        # short-circuiting the descriptor on the next access. This lets a
        # bare Mole stay lazy (pyscf_mol is seeded, hartree_fock is not) and
        # an already-converged mean field skip HF entirely (both are seeded).
        instance.pyscf_mol = mol
        if mf is not None:
            instance.hartree_fock = mf
        return instance

    @cached_property
    def pyscf_mol(self) -> gto.Mole:
        """PySCF Mole object."""
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
        mol = self.pyscf_mol
        pyscf_scf = scf.ROHF(mol) if mol.spin else scf.RHF(mol)
        if self.backend == "pyscf_mem_efficient":
            pyscf_scf.direct_scf = True
        elif self.backend == "pyscf_density_fitting":
            pyscf_scf = pyscf_scf.density_fit()
            pyscf_scf.with_df._cderi = df.incore.cholesky_eri(mol)
        pyscf_scf.run(verbose=0)
        if not pyscf_scf.converged:
            pyscf_scf = pyscf_scf.newton().run()
        if not pyscf_scf.converged:
            raise RuntimeError(f"PySCF Hartree-Fock failed to converge for {mol.atom}.")
        return pyscf_scf

    def get_hartree_fock(self) -> scf.hf.SCF:
        return self.hartree_fock

    @property
    def hf_energy(self) -> float:
        """Total Hartree-Fock energy (Hartree) of the underlying mean field."""
        return float(self.hartree_fock.e_tot)

    @cached_property
    def active_space(self) -> ActiveSpace:
        """Constructs the active space for the molecule.

        If this instance was built via :meth:`from_pyscf` with an explicit
        ``active_space``, that is returned as-is. Otherwise, the active
        space is defined by:
          - Excluding frozen orbitals if `self.frozen` is set.
          - Using the remaining orbitals as the active orbitals.
          - Number of active electrons is the total number of electrons
            minus twice the number of frozen orbitals.
          - Number of active orbitals is the total number of molecular orbitals
            minus the number of frozen orbitals.

        Returns:
            ActiveSpace: An `ActiveSpace` object describing the active orbitals and electrons.
        """
        if self._active_space_override is not None:
            return self._active_space_override
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

    @property
    def n_qubits(self) -> int:
        """Number of qubits the qubit Hamiltonian acts on (= 2 * active orbitals)."""
        return self.qubit_hamiltonian.n_qubit

    @property
    def n_electrons(self) -> int:
        """Number of active electrons."""
        return self.active_space.n_active_ele

    @cached_property
    def fermionic_hamiltonian(self) -> FermionicHamiltonian:
        as_eint_set, mo_eint_set = get_spin_mo_integrals_from_mole(
            self.pyscf_mol, self.hartree_fock.mo_coeff, self.active_space
        )
        fermion_op = get_fermionic_mapped_hamiltonian(mo_eint_set)
        n_spin_orbital = self.hartree_fock.mo_coeff.shape[1] * 2
        return FermionicHamiltonian(
            n_spin_orbital=n_spin_orbital,
            fermion_operator=cast(FermionOperator, fermion_op),
        )

    def get_fermionic_hamiltonian(self) -> FermionicHamiltonian:
        return self.fermionic_hamiltonian

    @cached_property
    def _qubit_op_and_mapping(self) -> tuple[Operator, OpenFermionQubitMapping]:
        return operator_from_of_fermionic_op(
            self.fermionic_hamiltonian.fermion_operator,
            self.active_space,
            sz=self.sz,
            fermion_qubit_mapping=self.fermion_qubit_mapping,
        )

    @cached_property
    def qubit_hamiltonian(self) -> QubitHamiltonian:
        """Return the qubit Hamiltonian, mapped via
        :attr:`fermion_qubit_mapping` (Jordan-Wigner by default)."""
        qubit_operator, mapping = self._qubit_op_and_mapping
        return QubitHamiltonian(
            n_qubit=mapping.n_qubits,
            qubit_hamiltonian=qubit_operator,
        )

    def get_qubit_hamiltonian(self) -> QubitHamiltonian:
        return self.qubit_hamiltonian

    @cached_property
    def hf_state(self) -> ComputationalBasisState:
        """Hartree-Fock reference as a computational basis state.

        Correct for any :attr:`fermion_qubit_mapping`: it is derived
        from that mapping's own state mapper rather than a mapping-
        specific bit convention, so it stays consistent with
        :attr:`qubit_hamiltonian` even for non-Jordan-Wigner mappings.
        """
        _, mapping = self._qubit_op_and_mapping
        active_space = self.active_space
        spin = int(self.pyscf_mol.spin)
        n_alpha = (active_space.n_active_ele + spin) // 2
        n_beta = (active_space.n_active_ele - spin) // 2
        occupied = [2 * i for i in range(n_alpha)] + [2 * i + 1 for i in range(n_beta)]
        return mapping.state_mapper(occupied)
