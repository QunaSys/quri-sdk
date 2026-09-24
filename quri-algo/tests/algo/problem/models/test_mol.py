# Licensed under the MIT License (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
from openfermion.ops.representations.interaction_operator import InteractionOperator
from pyscf import df, gto, scf
from pyscf.soscf.newton_ah import _CIAH_SOSCF
from quri_parts.chem.mol import ActiveSpace
from quri_parts.core.operator import get_sparse_matrix
from quri_parts.core.operator.operator import Operator
from quri_parts.openfermion.mol import (
    get_fermionic_hamiltonian,
    get_qubit_mapped_hamiltonian,
)
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    symmetry_conserving_bravyi_kitaev,
)
from quri_parts.pyscf.mol import get_spin_mo_integrals_from_mole

from quri_algo.problem.models.mol import MolecularSystem

H2O_COORDS = "O 0 0 0; H 0.2774 0.8929 0.2544; H 0.6068, -0.2383, -0.7169"


@pytest.fixture(scope="module")
def h2o_system() -> MolecularSystem:
    """Simple H2O molecule with minimal basis."""
    return MolecularSystem(atom=H2O_COORDS, basis="sto-3g", charge=0, spin=0)


def compare_ops(op1: Operator, op2: Operator, tol: float = 1e-12) -> bool:
    """Compare two qubit Hamiltonians by maximum absolute coefficient
    difference."""
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
    mol = MolecularSystem(atom=H2O_COORDS, frozen=frozen)
    ref_h = reference_qubit_hamiltonian(H2O_COORDS, frozen=frozen)
    qh = mol.get_qubit_hamiltonian()
    assert compare_ops(qh.qubit_hamiltonian, ref_h)


def test_qubit_hamiltonian_differences() -> None:
    """Test that freezing orbitals changes the qubit Hamiltonian."""
    mol_frozen = MolecularSystem(atom=H2O_COORDS, frozen=[0])
    mol_full = MolecularSystem(atom=H2O_COORDS)
    qh_frozen = mol_frozen.get_qubit_hamiltonian()
    qh_full = mol_full.get_qubit_hamiltonian()
    assert not compare_ops(qh_frozen.qubit_hamiltonian, qh_full.qubit_hamiltonian)


def compare_interaction_operators(
    op1: InteractionOperator, op2: InteractionOperator, tol: float = 1e-12
) -> bool:
    """Compare two InteractionOperator objects for approximate equality."""

    if op1.n_qubits != op2.n_qubits:
        return False

    if not np.isclose(op1.constant, op2.constant, atol=tol):
        return False

    if not np.allclose(op1.one_body_tensor, op2.one_body_tensor, atol=tol):
        return False

    if not np.allclose(op1.two_body_tensor, op2.two_body_tensor, atol=tol):
        return False

    return True


def reference_fermionic_hamiltonian(
    atom_coords: str, frozen: list[int] | None = None
) -> InteractionOperator:
    """Compute reference Fermionic Hamiltonian from PySCF."""

    mole = gto.M(atom=atom_coords)
    mf = scf.RHF(mole).run(verbose=0)
    active_space = ActiveSpace(8, 6) if frozen else None
    _, cas_mo_eint_set = get_spin_mo_integrals_from_mole(
        mole, mf.mo_coeff, active_space=active_space
    )
    ref_h = get_fermionic_hamiltonian(cas_mo_eint_set)
    return ref_h


@pytest.mark.parametrize("frozen", [None, [0]])
def test_fermionic_hamiltonian_matches_reference(frozen: list[int] | None) -> None:
    """Test MolecularSystem qubit Hamiltonian against reference computation."""
    mol = MolecularSystem(atom=H2O_COORDS, frozen=frozen)
    ref_h = reference_fermionic_hamiltonian(H2O_COORDS, frozen=frozen)
    fh = mol.get_fermionic_hamiltonian()
    assert compare_interaction_operators(
        cast(InteractionOperator, fh.fermion_operator), ref_h
    )


def test_fermionic_hamiltonian_differences() -> None:
    """Test that freezing orbitals changes the qubit Hamiltonian."""
    mol_frozen = MolecularSystem(atom=H2O_COORDS, frozen=[0])
    mol_full = MolecularSystem(atom=H2O_COORDS)
    fh_frozen = cast(
        InteractionOperator, mol_frozen.get_fermionic_hamiltonian().fermion_operator
    )
    fh_full = cast(
        InteractionOperator, mol_full.get_fermionic_hamiltonian().fermion_operator
    )
    assert not compare_interaction_operators(fh_frozen, fh_full)


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
    sys = MolecularSystem(atom=H2O_COORDS, basis="sto-3g", backend=backend)
    mf = sys.get_hartree_fock()
    assert mf.converged

    mol = gto.M(atom=H2O_COORDS, basis="sto-3g", unit="Angstrom")
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


H2_COORDS = "H 0 0 0; H 0 0 0.74"


def test_from_pyscf_bare_mole_is_lazy() -> None:
    """Constructing from_pyscf on a bare Mole must not run HF up front."""
    mol = MolecularSystem.from_pyscf(gto.M(atom=H2_COORDS, basis="sto-3g"))
    assert "hartree_fock" not in mol.__dict__
    assert mol.active_space.n_active_orb == 2  # doesn't need HF
    assert "hartree_fock" not in mol.__dict__
    assert mol.hf_energy < 0  # now it runs
    assert "hartree_fock" in mol.__dict__


def test_from_pyscf_converged_mf_is_reused_not_rerun() -> None:
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g")).run(verbose=0)
    with patch.object(mf, "kernel") as kernel:
        mol = MolecularSystem.from_pyscf(mf)
        assert mol.hartree_fock is mf
    kernel.assert_not_called()


def _cap_rhf_max_cycle(monkeypatch: pytest.MonkeyPatch, newton_max_cycle: int) -> None:
    """Leave the initial RHF unconverged and cap its Newton fallback.

    The initial RHF in ``hartree_fock`` gets ``max_cycle=0``; its Newton
    fallback gets ``newton_max_cycle`` iterations.
    """
    orig_rhf = scf.RHF

    def capped_rhf(mol: gto.Mole) -> scf.hf.SCF:
        mf = orig_rhf(mol)
        mf.max_cycle = 0
        orig_newton = mf.newton

        def newton() -> scf.hf.SCF:
            newton_mf = orig_newton()
            newton_mf.max_cycle = newton_max_cycle
            return newton_mf

        mf.newton = newton
        return mf

    monkeypatch.setattr(scf, "RHF", capped_rhf)


def test_hartree_fock_newton_fallback_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cap_rhf_max_cycle(monkeypatch, newton_max_cycle=50)
    mol = MolecularSystem(atom=H2_COORDS, basis="sto-3g")
    mf = mol.hartree_fock
    assert mf.converged
    assert isinstance(mf, _CIAH_SOSCF)
    assert mol.hartree_fock is mf


def test_hartree_fock_newton_fallback_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _cap_rhf_max_cycle(monkeypatch, newton_max_cycle=0)
    mol = MolecularSystem(atom=H2_COORDS, basis="sto-3g")
    with pytest.raises(RuntimeError):
        mol.hartree_fock


def test_from_pyscf_fewer_mos_than_aos(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default active space must follow the mean field's MO count, which
    linear-dependence removal can make smaller than ``mol.nao``."""
    monkeypatch.setattr(scf.hf, "remove_overlap_zero_eigenvalue", True)
    monkeypatch.setattr(scf.hf, "overlap_zero_eigenvalue_threshold", 0.5)
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="6-31g")).run(verbose=0)
    assert mf.mol.nao == 4
    assert mf.mo_coeff.shape == (4, 2)
    mol = MolecularSystem.from_pyscf(mf)
    assert mol.active_space.n_active_orb == 2
    assert mol.n_qubits == 4
    assert mol.hf_state.bits == 0b0011


def test_from_pyscf_hf_state_rejects_fractional_occupation() -> None:
    mf = scf.addons.smearing_(
        scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g")), sigma=0.5
    ).run(verbose=0)
    mol = MolecularSystem.from_pyscf(mf)
    with pytest.raises(ValueError, match="occupation"):
        mol.hf_state


@pytest.mark.parametrize(
    "atom, spin, sz, expected_bits",
    [
        (H2_COORDS, 0, 0.0, 0b0011),  # RHF: doubly occupied
        ("H 0 0 0", 1, 0.5, 0b01),  # ROHF: single alpha
        ("H 0 0 0", -1, -0.5, 0b10),  # ROHF: single beta
    ],
)
def test_from_pyscf_hf_state_integer_occupations(
    atom: str, spin: int, sz: float, expected_bits: int
) -> None:
    mole = gto.M(atom=atom, basis="sto-3g", spin=spin)
    mf = (scf.ROHF(mole) if spin else scf.RHF(mole)).run(verbose=0)
    for mol in (
        MolecularSystem.from_pyscf(mf),
        MolecularSystem.from_pyscf(mf, sz=sz),
    ):
        assert mol.hf_state.bits == expected_bits


def test_from_pyscf_matches_default_constructor() -> None:
    """The two constructors must converge on the same Hamiltonian/HF state."""
    default = MolecularSystem(atom=H2_COORDS, basis="sto-3g")
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g")).run(verbose=0)
    wrapped = MolecularSystem.from_pyscf(mf)
    assert compare_ops(
        default.qubit_hamiltonian.qubit_hamiltonian,
        wrapped.qubit_hamiltonian.qubit_hamiltonian,
    )
    assert default.hf_state.bits == wrapped.hf_state.bits == 0b0011


def test_from_pyscf_rejects_unconverged_mf() -> None:
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g"))
    mf.max_cycle = 0  # force non-convergence
    mf.run(verbose=0)
    with pytest.raises(RuntimeError):
        MolecularSystem.from_pyscf(mf)


def test_from_pyscf_rejects_uhf() -> None:
    mf = scf.UHF(gto.M(atom=H2_COORDS, basis="sto-3g")).run(verbose=0)
    with pytest.raises(NotImplementedError):
        MolecularSystem.from_pyscf(mf)


def test_from_pyscf_active_space_override() -> None:
    mf = scf.RHF(gto.M(atom=H2O_COORDS, basis="sto-3g")).run(verbose=0)
    active_space = ActiveSpace(8, 6)
    mol = MolecularSystem.from_pyscf(mf, active_space=active_space)
    assert mol.active_space is active_space
    ref_h = reference_qubit_hamiltonian(H2O_COORDS, frozen=[0])
    assert compare_ops(mol.qubit_hamiltonian.qubit_hamiltonian, ref_h)


def test_from_pyscf_hf_state_reordered_active_orbitals() -> None:
    """hf_state must follow the mean field's actual occupation, not the active
    orbital index order (issue #952)."""
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g")).run(verbose=0)
    mol = MolecularSystem.from_pyscf(mf, ActiveSpace(2, 2, [1, 0]))
    assert mol.hf_state.bits == 0b1100
    mat = get_sparse_matrix(mol.qubit_hamiltonian.qubit_hamiltonian).toarray()
    hf_state_energy = mat[mol.hf_state.bits, mol.hf_state.bits].real
    assert np.isclose(hf_state_energy, mol.hf_energy, atol=1e-8)


def test_from_pyscf_hf_state_permuted_mean_field() -> None:
    """hf_state must follow mo_occ of a permuted mean field (issue #952).

    Same bug as above without an active-space override: mo_coeff,
    mo_occ, and mo_energy are consistently permuted.
    """
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g")).run(verbose=0)
    perm = [1, 0]
    mf.mo_coeff = mf.mo_coeff[:, perm]
    mf.mo_occ = mf.mo_occ[perm]
    mf.mo_energy = mf.mo_energy[perm]
    mol = MolecularSystem.from_pyscf(mf)
    assert mol.hf_state.bits == 0b1100
    mat = get_sparse_matrix(mol.qubit_hamiltonian.qubit_hamiltonian).toarray()
    hf_state_energy = mat[mol.hf_state.bits, mol.hf_state.bits].real
    assert np.isclose(hf_state_energy, mol.hf_energy, atol=1e-8)


def test_from_pyscf_hf_state_matches_mapping_for_non_jw() -> None:
    """hf_state must be derived from the requested mapping's own state mapper,
    not a Jordan-Wigner-shaped bit convention -- this is the bug the from_pyscf
    helper this replaces had (see PR #297)."""
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g")).run(verbose=0)
    mol = MolecularSystem.from_pyscf(mf, fermion_qubit_mapping=bravyi_kitaev)
    _, mapping = mol._qubit_op_and_mapping
    expected = mapping.state_mapper([0, 1])
    assert mol.hf_state.bits == expected.bits

    jw = MolecularSystem.from_pyscf(mf)
    bk_gse = np.linalg.eigvalsh(
        get_sparse_matrix(mol.qubit_hamiltonian.qubit_hamiltonian).toarray()
    )[0]
    jw_gse = np.linalg.eigvalsh(
        get_sparse_matrix(jw.qubit_hamiltonian.qubit_hamiltonian).toarray()
    )[0]
    assert abs(bk_gse - jw_gse) < 1e-8  # same physics, different mapping


def test_from_pyscf_n_qubits_matches_reduced_mapping() -> None:
    """``n_qubits`` must reflect the mapping's own qubit count (issue #953).

    Not 2 * n_active_orb: symmetry_conserving_bravyi_kitaev drops two
    qubits, and qubit_hamiltonian.n_qubit/hf_state must agree.
    """
    mf = scf.RHF(gto.M(atom=H2_COORDS, basis="sto-3g")).run(verbose=0)
    mol = MolecularSystem.from_pyscf(
        mf, fermion_qubit_mapping=symmetry_conserving_bravyi_kitaev, sz=0
    )
    assert mol.n_qubits == 2
    assert mol.qubit_hamiltonian.n_qubit == 2
    assert mol.hf_state.qubit_count == 2
