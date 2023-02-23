# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, cast

import numpy as np
from pyscf.gto import Mole

from quri_parts.chem.mol import MolecularOrbitals

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


class PySCFMolecularOrbitals(MolecularOrbitals):
    """PySCFMolecularOrbitals represents a data of the molecule which consists
    of the PySCF Mole object :class:`Mole` and the molecular orbital
    coefficients.

    Args:
        mol: :class:`Mole` the PySCF Mole object for the molecule.
        mo_coeff: molecular orbital coefficients.
    """

    def __init__(self, mol: Mole, mo_coeff: "npt.NDArray[np.complex128]"):
        self._mol = mol
        self._mo_coeff = mo_coeff

    @property
    def mol(self) -> Mole:
        """Returns :class:`Mole` the PySCF Mole object of the molecule."""
        return self._mol

    @property
    def n_electron(self) -> int:
        """Returns a number of electrons."""
        return cast(int, self._mol.nelectron)

    @property
    def mo_coeff(self) -> "npt.NDArray[np.complex128]":
        """Returns molecular orbital coefficients."""
        return self._mo_coeff


def get_nuc_energy(mo: PySCFMolecularOrbitals) -> float:
    """Returns a nuclear repulsion energy of a given molecule.

    Args:
        mo: :class:`PySCFMolecularOrbitals` of the molecule.
    """

    return cast(float, mo.mol.energy_nuc())
