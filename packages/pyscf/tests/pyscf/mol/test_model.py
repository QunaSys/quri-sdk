# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyscf import gto, scf
from pyscf.gto import Mole

from quri_parts.pyscf.mol.model import PySCFMolecularOrbitals, get_nuc_energy

mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
mf = scf.HF(mol)
mf.kernel()
test_mol_set = PySCFMolecularOrbitals(mol, mf.mo_coeff)


class TestPySCFMolecularOrbitals:
    def test_mol(self) -> None:
        test_mol = test_mol_set.mol
        assert isinstance(test_mol, Mole)
        assert test_mol.basis == "sto-3g"
        assert test_mol.atom == "H 0 0 0; H 0 0 1"

    def test_n_electron(self) -> None:
        assert test_mol_set.n_electron == 2

    def test_spin(self) -> None:
        assert test_mol_set.spin == 0

    def test_n_spatial_orb(self) -> None:
        assert test_mol_set.n_spatial_orb == 2

    def test_mo_coeff(self) -> None:
        test_mo_coeff = test_mol_set.mo_coeff
        assert test_mo_coeff.shape == (2, 2)


def test_get_nuc_energy() -> None:
    nuc_energy = get_nuc_energy(test_mol_set)

    assert nuc_energy == test_mol_set.mol.energy_nuc()
