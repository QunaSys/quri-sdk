# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Union, cast

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo, gto, mcscf, scf

from quri_parts.chem.mol import (
    ActiveSpace,
    ActiveSpaceMolecularOrbitals,
    AO1eInt,
    AO1eIntArray,
    AO2eInt,
    AO2eIntArray,
    AOeIntArraySet,
    AOeIntSet,
    MolecularOrbitals,
    SpatialMO1eInt,
    SpatialMO1eIntArray,
    SpatialMO2eInt,
    SpatialMO2eIntArray,
    SpatialMOeIntSet,
    SpinMO1eInt,
    SpinMO2eInt,
    SpinMOeIntSet,
    spatial_mo_1e_int_to_spin_mo_1e_int,
    spatial_mo_2e_int_to_spin_mo_2e_int,
    spatial_mo_eint_set_to_spin_mo_eint_set,
)

from .model import PySCFMolecularOrbitals, get_nuc_energy


class PySCFAO1eInt(AO1eInt):
    def __init__(self, mol: gto.Mole) -> None:
        self._mol = mol

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the ao 1-electron integrals with pyscf."""
        ao_int1e = scf.hf.get_hcore(self._mol)
        return cast(npt.NDArray[np.complex128], ao_int1e)

    def to_spatial_mo1int(
        self, mo_coeff: "npt.NDArray[np.complex128]"
    ) -> SpatialMO1eInt:
        """Returns the corresponding spatial mo 1e integrals."""
        return PySCFSpatialMO1eInt(mol=self._mol, mo_coeff=mo_coeff)

    def to_mo1int(self, mo_coeff: "npt.NDArray[np.complex128]") -> SpinMO1eInt:
        """Returns the corresponding spin mo 1e integrals."""
        return PySCFSpinMO1eInt(mol=self._mol, mo_coeff=mo_coeff)


class PySCFAO2eInt(AO2eInt):
    def __init__(self, mol: gto.Mole) -> None:
        self._mol = mol

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the ao 2-electron integrals in a memory effecient way, and
        returns the array in physicist's notation."""
        ao_int2e = ao2mo.get_ao_eri(mol=self._mol)
        ao_int2e = ao2mo.restore(1, ao_int2e, self._mol.nao).transpose(0, 2, 3, 1)
        return cast(npt.NDArray[np.complex128], ao_int2e)

    def to_spatial_mo2int(
        self, mo_coeff: "npt.NDArray[np.complex128]"
    ) -> SpatialMO2eInt:
        """Returns the corresponding spatial mo 2e integrals."""
        return PySCFSpatialMO2eInt(mol=self._mol, mo_coeff=mo_coeff)

    def to_mo2int(self, mo_coeff: "npt.NDArray[np.complex128]") -> SpinMO2eInt:
        """Returns the corresponding spin mo 2e integrals."""
        return PySCFSpinMO2eInt(mol=self._mol, mo_coeff=mo_coeff)


@dataclass
class PySCFAOeIntSet(AOeIntSet):
    mol: gto.Mole
    constant: float
    ao_1e_int: PySCFAO1eInt
    ao_2e_int: PySCFAO2eInt

    def to_full_space_mo_int(
        self,
        mo: MolecularOrbitals,
    ) -> SpinMOeIntSet:
        """Returns the full space spin electon integrals."""
        spin_mo_eint_set = SpinMOeIntSet(
            const=self.constant,
            mo_1e_int=self.ao_1e_int.to_mo1int(mo.mo_coeff),
            mo_2e_int=self.ao_2e_int.to_mo2int(mo.mo_coeff),
        )
        return spin_mo_eint_set

    def to_full_space_spatial_mo_int(
        self,
        mo: MolecularOrbitals,
    ) -> SpatialMOeIntSet:
        """Returns the full space spatial electon integrals."""
        spatial_mo_eint_set = SpatialMOeIntSet(
            const=self.constant,
            mo_1e_int=self.ao_1e_int.to_spatial_mo1int(mo.mo_coeff),
            mo_2e_int=self.ao_2e_int.to_spatial_mo2int(mo.mo_coeff),
        )
        return spatial_mo_eint_set

    def to_active_space_mo_int(
        self,
        active_space_mo: ActiveSpaceMolecularOrbitals,
    ) -> SpinMOeIntSet:
        """Returns the full space spin electon integrals."""
        spin_mo_eint_set = get_active_space_spin_integrals(active_space_mo, self)
        return spin_mo_eint_set

    def to_active_space_spatial_mo_int(
        self,
        active_space_mo: ActiveSpaceMolecularOrbitals,
    ) -> SpatialMOeIntSet:
        """Returns the full space spatial electon integrals."""
        spatial_mo_eint_set = get_active_space_spatial_integrals(active_space_mo, self)
        return spatial_mo_eint_set


class PySCFSpatialMO1eInt(SpatialMO1eInt):
    def __init__(self, mol: gto.Mole, mo_coeff: npt.NDArray[np.complex128]) -> None:
        self._mol = mol
        self._mo_coeff = mo_coeff

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the mo 1-electron integrals and returns the array in
        physicist's notation."""
        h1_mo = (
            self._mo_coeff.conjugate().T @ scf.hf.get_hcore(self._mol) @ self._mo_coeff
        )
        return cast(npt.NDArray[np.complex128], h1_mo)


class PySCFSpatialMO2eInt(SpatialMO2eInt):
    def __init__(self, mol: gto.Mole, mo_coeff: npt.NDArray[np.complex128]) -> None:
        self._mol = mol
        self._mo_coeff = mo_coeff

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Computes the mo 2-electron integrals in a memory effecient way, and
        returns the array in physicist's notation."""
        # Pass in Mole object to the eri_or_mol kwarg for memory efficiency
        # The ao2mo.get_mo_eri function uses ao2mo.outcore to
        # evaluate the matrix elements
        mo_int_2e = ao2mo.get_mo_eri(eri_or_mol=self._mol, mo_coeffs=self._mo_coeff)
        mo_int_2e = ao2mo.restore(1, mo_int_2e, self._mol.nao)
        mo_int_2e = mo_int_2e.transpose(0, 2, 3, 1)
        return cast(npt.NDArray[np.complex128], mo_int_2e)


class PySCFSpinMO1eInt(SpinMO1eInt):
    def __init__(self, mol: gto.Mole, mo_coeff: npt.NDArray[np.complex128]) -> None:
        self._mol = mol
        self._mo_coeff = mo_coeff

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        spatial_1e_int = PySCFSpatialMO1eInt(
            mol=self._mol, mo_coeff=self._mo_coeff
        ).array
        n_spin_orb = spatial_1e_int.shape[0] * 2
        spin_1e_int = spatial_mo_1e_int_to_spin_mo_1e_int(n_spin_orb, spatial_1e_int)
        return spin_1e_int


class PySCFSpinMO2eInt(SpinMO2eInt):
    def __init__(self, mol: gto.Mole, mo_coeff: npt.NDArray[np.complex128]) -> None:
        self._mol = mol
        self._mo_coeff = mo_coeff

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        spatial_2e_int = PySCFSpatialMO2eInt(
            mol=self._mol, mo_coeff=self._mo_coeff
        ).array
        n_spin_orb = spatial_2e_int.shape[0] * 2
        spin_2e_int = spatial_mo_2e_int_to_spin_mo_2e_int(n_spin_orb, spatial_2e_int)
        return spin_2e_int


def get_ao_1eint(mo: PySCFMolecularOrbitals) -> PySCFAO1eInt:
    """Returns a PySCFAO1eInt object."""
    return PySCFAO1eInt(mol=mo.mol)


def get_ao_2eint(mo: PySCFMolecularOrbitals) -> PySCFAO2eInt:
    """Returns a PySCFAO2eInt object."""
    return PySCFAO2eInt(mol=mo.mol)


def get_ao_eint_set(
    molecule: PySCFMolecularOrbitals, store_array_on_memory: bool = False
) -> Union[PySCFAOeIntSet, AOeIntArraySet]:
    """Returns an instance of either PySCFAOeIntSet or AOeIntArraySet.

    If store_array_on_memory == True, it computes the ao electron integrals
    and returns AOeIntArraySet that holds the electron integral arrays on
    memory.

    If store_array_on_memory == False, it returns a PySCFAOeIntSet object
    that only holds the pyscf.gto.Mole object on memory.
    """
    nuc_energy = get_nuc_energy(molecule)
    ao_1e_int = get_ao_1eint(molecule)
    ao_2e_int = get_ao_2eint(molecule)

    if store_array_on_memory:
        ao_1e_int_array = AO1eIntArray(ao_1e_int.array)
        ao_2e_int_array = AO2eIntArray(ao_2e_int.array)
        ao_e_int_array_set = AOeIntArraySet(
            constant=nuc_energy,
            ao_1e_int=ao_1e_int_array,
            ao_2e_int=ao_2e_int_array,
        )
        return ao_e_int_array_set

    ao_e_int_set = PySCFAOeIntSet(
        mol=molecule.mol,
        constant=nuc_energy,
        ao_1e_int=ao_1e_int,
        ao_2e_int=ao_2e_int,
    )
    return ao_e_int_set


def get_active_space_spatial_integrals(
    active_space_mo: ActiveSpaceMolecularOrbitals,
    electron_ints: PySCFAOeIntSet,
) -> SpatialMOeIntSet:
    """Computes the active space electron integrals by pyscf.casci.

    The output is the spatial mo electron integrals in physicists'
    convention.
    """
    n_active_orb, n_active_ele = (
        active_space_mo.n_active_orb,
        active_space_mo.n_active_ele,
    )
    casci = mcscf.CASCI(electron_ints.mol, n_active_orb, n_active_ele)
    _, active_orb_list = active_space_mo.get_core_and_active_orb()
    if active_orb_list:
        mo = casci.sort_mo(active_orb_list, mo_coeff=active_space_mo.mo_coeff, base=0)
    else:
        mo = active_space_mo.mo_coeff

    casci_mo_1e_int, casci_nuc = casci.get_h1eff(mo)
    casci_mo_2e_int = casci.get_h2eff(mo)
    casci_mo_2e_int = ao2mo.restore(1, casci_mo_2e_int, casci.ncas).transpose(
        0, 2, 3, 1
    )

    spatial_integrals = SpatialMOeIntSet(
        const=casci_nuc,
        mo_1e_int=SpatialMO1eIntArray(casci_mo_1e_int),
        mo_2e_int=SpatialMO2eIntArray(casci_mo_2e_int),
    )

    return spatial_integrals


def get_active_space_spin_integrals(
    active_space_mo: ActiveSpaceMolecularOrbitals,
    electron_ints: PySCFAOeIntSet,
) -> SpinMOeIntSet:
    """Computes the active space electron integrals by pyscf.casci.

    The output is the spin mo electron integrals in physicists'
    convention.
    """
    spatial_integrals = get_active_space_spatial_integrals(
        active_space_mo,
        electron_ints,
    )
    spin_integrals = spatial_mo_eint_set_to_spin_mo_eint_set(spatial_integrals)
    return spin_integrals


def get_spin_mo_integrals_from_mole(
    mole: gto.Mole,
    mo_coeff: npt.NDArray[np.complex128],
    active_space: Optional[ActiveSpace] = None,
) -> tuple[ActiveSpace, SpinMOeIntSet]:
    """Computes the spin MO electron integrals and the corresponding
    :class:`~quri_parts.chem.mol.ActiveSpace` object."""
    mo = PySCFMolecularOrbitals(mole, mo_coeff)
    ao_eint_set = get_ao_eint_set(mo)

    if active_space is None:
        return (
            ActiveSpace(mo.n_electron, mo.n_spatial_orb),
            ao_eint_set.to_full_space_mo_int(mo),
        )

    asmo = ActiveSpaceMolecularOrbitals(mo, active_space)
    return active_space, ao_eint_set.to_active_space_mo_int(asmo)
