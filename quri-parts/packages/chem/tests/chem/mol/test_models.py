# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

import numpy as np

from quri_parts.chem.mol import (
    ActiveSpace,
    ActiveSpaceMolecularOrbitals,
    MolecularOrbitals,
)
from quri_parts.chem.mol.models import OrbitalType

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


test_mo_coeff = np.array([[i for i in range(6)] for j in range(6)])
cas = ActiveSpace(2, 4, [4, 5, 6, 7])


class TestMolecularOrbitals(MolecularOrbitals):
    @property
    def n_electron(self) -> int:
        return 8

    @property
    def n_spatial_orb(self) -> int:
        return 20

    @property
    def spin(self) -> int:
        return 2

    @property
    def mo_coeff(self) -> "npt.NDArray[np.complex128]":
        return test_mo_coeff


class TestActiveSpaceMolecularOrbitals:
    test_mol_set = TestMolecularOrbitals()
    as_mol = ActiveSpaceMolecularOrbitals(test_mol_set, cas)

    def test_n_electron(self) -> None:
        assert self.as_mol.n_electron == 8

    def test_mo_coeff(self) -> None:
        assert np.allclose(self.as_mol.mo_coeff, test_mo_coeff)

    def test_get_core_and_active_orb(self) -> None:
        assert self.as_mol.get_core_and_active_orb() == ([0, 1, 2], [4, 5, 6, 7])

    def test_orb_type(self) -> None:
        assert self.as_mol.orb_type(1) == OrbitalType.CORE
        assert self.as_mol.orb_type(3) == OrbitalType.VIRTUAL
        assert self.as_mol.orb_type(5) == OrbitalType.ACTIVE

    def test_spin(self) -> None:
        assert self.as_mol.spin == 2

    def test_n_active_ele(self) -> None:
        assert self.as_mol.n_active_ele == 2

    def test_n_core_ele(self) -> None:
        assert self.as_mol.n_core_ele == 6

    def test_n_ele_alpha(self) -> None:
        assert self.as_mol.n_ele_alpha == 2

    def test_n_ele_beta(self) -> None:
        assert self.as_mol.n_ele_beta == 0

    def test_n_spatial_orb(self) -> None:
        assert self.as_mol.n_spatial_orb == 20

    def test_n_active_orb(self) -> None:
        assert self.as_mol.n_active_orb == 4

    def test_n_core_orb(self) -> None:
        assert self.as_mol.n_core_orb == 3

    def test_n_vir_orb(self) -> None:
        assert self.as_mol.n_vir_orb == 20 - 4 - 3
