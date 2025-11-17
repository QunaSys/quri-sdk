# Licensed under the MIT License (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

import numpy as np
import pytest
from pyscf import df, gto, scf
from quri_parts.chem.mol import ActiveSpace
from quri_parts.core.operator.operator import Operator
from quri_parts.openfermion.mol import get_qubit_mapped_hamiltonian
from quri_parts.pyscf.mol import get_spin_mo_integrals_from_mole

from quri_algo.problem.models.mol import MolecularSystem

H2O_COORDS = "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0"
H2O_TEST_COORDS = "O 0 0 0; H 0.2774 0.8929 0.2544; H 0.6068, -0.2383, -0.7169"


@pytest.fixture(scope="module")
def h2o_system() -> MolecularSystem:
    """Simple H2O molecule with minimal basis."""
    return MolecularSystem(atom=H2O_COORDS, basis="sto-3g", charge=0, spin=0)


def compare_ops(op1: Operator, op2: Operator, tol: float = 1e-12) -> bool:
    """Compare two qubit Hamiltonians by maximum absolute coefficient difference."""
    diff = op1 - op2
    if not diff:  # empty difference → operators are identical
        return True
    max_abs_diff = max(abs(c) for c in diff.values())
    return bool(max_abs_diff <= tol)


def reference_qubit_hamiltonian(
    atom_coords: str, frozen: list[int] | None = None
) -> Operator:
    """Compute reference JW-mapped qubit Hamiltonian from PySCF."""
    mole = gto.M(atom=atom_coords)
    mf = scf.RHF(mole).run(verbose=0)
    active_space = ActiveSpace(8, 6) if frozen else None
    ref_h, _ = get_qubit_mapped_hamiltonian(
        *get_spin_mo_integrals_from_mole(mole, mf.mo_coeff, active_space=active_space)
    )
    return ref_h


@pytest.mark.parametrize("frozen", [None, [0]])
def test_qubit_hamiltonian_matches_reference(frozen: list[int] | None) -> None:
    """Test MolecularSystem qubit Hamiltonian against reference computation."""
    mol = MolecularSystem(atom=H2O_TEST_COORDS, frozen=frozen)
    ref_h = reference_qubit_hamiltonian(H2O_TEST_COORDS, frozen=frozen)
    qh = mol.get_qubit_hamiltonian()
    assert compare_ops(qh.qubit_hamiltonian, ref_h)


def test_qubit_hamiltonian_differences() -> None:
    """Test that freezing orbitals changes the qubit Hamiltonian."""
    mol_frozen = MolecularSystem(atom=H2O_TEST_COORDS, frozen=[0])
    mol_full = MolecularSystem(atom=H2O_TEST_COORDS)
    qh_frozen = mol_frozen.get_qubit_hamiltonian()
    qh_full = mol_full.get_qubit_hamiltonian()
    assert not compare_ops(qh_frozen.qubit_hamiltonian, qh_full.qubit_hamiltonian)


def test_cached_attributes(h2o_system: MolecularSystem) -> None:
    """Test that MolecularSystem caching works correctly."""
    mol = h2o_system

    # Ensure all cached getters return objects
    assert mol.qubit_hamiltonian is not None
    assert mol.fermionic_hamiltonian is not None
    assert mol.active_space is not None
    assert mol.hartree_fock is not None
    assert mol.pyscf_mol is not None

    # Ensure repeated calls return cached objects
    assert mol.qubit_hamiltonian is mol.qubit_hamiltonian
    assert mol.fermionic_hamiltonian is mol.fermionic_hamiltonian
    assert mol.active_space is mol.active_space
    assert mol.hartree_fock is mol.hartree_fock
    assert mol.pyscf_mol is mol.pyscf_mol


@pytest.mark.parametrize("backend", ["pyscf_mem_efficient", "pyscf_density_fitting"])
def test_hartree_fock_matches_pyscf_and_backend(
    backend: Literal["pyscf_mem_efficient", "pyscf_density_fitting"]
) -> None:
    sys = MolecularSystem(atom=H2O_TEST_COORDS, basis="sto-3g", backend=backend)
    mf = sys.get_hartree_fock()
    assert mf.converged

    mol = gto.M(atom=H2O_TEST_COORDS, basis="sto-3g", unit="Angstrom")
    mf_ref = scf.ROHF(mol) if mol.spin else scf.RHF(mol)
    if backend == "pyscf_mem_efficient":
        mf_ref.direct_scf = True
    elif backend == "pyscf_density_fitting":
        mf_ref = mf_ref.density_fit()
        mf_ref.with_df._cderi = df.incore.cholesky_eri(mol)
    mf_ref.run(verbose=0)

    assert mf_ref.converged
    for i in range(mf_ref.mo_coeff.shape[1]):
        col1 = mf_ref.mo_coeff[:, i]
        col2 = mf.mo_coeff[:, i]
        assert np.allclose(col1, col2, atol=1e-12) or np.allclose(
            col1, -col2, atol=1e-12
        )
    assert np.allclose(mf_ref.e_tot, mf.e_tot, atol=1e-12)


def test_active_space_default(h2o_system: MolecularSystem) -> None:
    aspace = h2o_system.get_active_space()
    assert aspace.n_active_ele == 10
    assert aspace.n_active_orb == 7
    assert h2o_system.active_space is aspace


def test_active_space_with_frozen() -> None:
    mol = MolecularSystem(atom=H2O_COORDS, frozen=[0])
    aspace = mol.active_space  # property auto-builds & caches
    assert isinstance(aspace, ActiveSpace)
    assert aspace.n_active_ele == 8
    assert aspace.n_active_orb == 6
    assert mol.active_space is aspace  # cached
