# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from numpy import allclose, array, complex128, isclose
from numpy.typing import NDArray
from pyscf import gto, scf

from quri_parts.chem.mol import ActiveSpace, ActiveSpaceMolecularOrbitals, cas
from quri_parts.pyscf.mol import (
    PySCFAO1eInt,
    PySCFAO2eInt,
    PySCFAOeIntSet,
    PySCFMolecularOrbitals,
    PySCFSpatialMO1eInt,
    PySCFSpatialMO2eInt,
    get_active_space_spatial_integrals,
    get_spin_mo_integrals_from_mole,
)


class TestH3WithMultiplicity2(unittest.TestCase):
    gto_mol: gto.mole.Mole
    mo_coeff: NDArray[complex128]
    pyscf_mol: PySCFMolecularOrbitals

    @classmethod
    def setUpClass(cls) -> None:
        atoms = [["H", [0, 0, i]] for i in range(3)]
        cls.gto_mol = gto.M(atom=atoms, spin=1)
        mf = scf.RHF(cls.gto_mol)
        mf.kernel()

        cls.mo_coeff = mf.mo_coeff
        cls.pyscf_mol = PySCFMolecularOrbitals(cls.gto_mol, cls.mo_coeff)

    def test_ao_1e_int(self) -> None:
        pyscf_ao_1e_int = PySCFAO1eInt(self.gto_mol)
        expected_ao_1e_int = array(
            [
                [-1.24398564, -0.86080698, -0.22631877],
                [-0.86080698, -1.4924109, -0.86080698],
                [-0.22631877, -0.86080698, -1.24398564],
            ]
        )
        assert allclose(pyscf_ao_1e_int.array, expected_ao_1e_int)

    def test_spatial_mo_1e_int(self) -> None:
        pyscf_spatial_mo_1e_int = PySCFSpatialMO1eInt(self.gto_mol, self.mo_coeff)
        expected_spatial_mo_1e_int = array(
            [
                [-1.53466912, -0.0, -0.12053411],
                [0.0, -1.15541096, -0.0],
                [-0.12053411, -0.0, -0.75066168],
            ]
        )
        assert allclose(pyscf_spatial_mo_1e_int.array, expected_spatial_mo_1e_int)

    def test_ao_2e_int(self) -> None:
        pyscf_ao_2e_int = PySCFAO2eInt(self.gto_mol)
        expected_ao_2e_int = array(
            [
                [
                    [
                        [0.77460594, 0.30930897, 0.05584872],
                        [0.30930897, 0.15786578, 0.03152398],
                        [0.05584872, 0.03152398, 0.00732843],
                    ],
                    [
                        [0.30930897, 0.15786578, 0.03152398],
                        [0.47804137, 0.30930897, 0.07065588],
                        [0.17348403, 0.11521591, 0.03152398],
                    ],
                    [
                        [0.05584872, 0.03152398, 0.00732843],
                        [0.17348403, 0.11521591, 0.03152398],
                        [0.26369505, 0.17348403, 0.05584872],
                    ],
                ],
                [
                    [
                        [0.30930897, 0.47804137, 0.17348403],
                        [0.15786578, 0.30930897, 0.11521591],
                        [0.03152398, 0.07065588, 0.03152398],
                    ],
                    [
                        [0.15786578, 0.30930897, 0.11521591],
                        [0.30930897, 0.77460594, 0.30930897],
                        [0.11521591, 0.30930897, 0.15786578],
                    ],
                    [
                        [0.03152398, 0.07065588, 0.03152398],
                        [0.11521591, 0.30930897, 0.15786578],
                        [0.17348403, 0.47804137, 0.30930897],
                    ],
                ],
                [
                    [
                        [0.05584872, 0.17348403, 0.26369505],
                        [0.03152398, 0.11521591, 0.17348403],
                        [0.00732843, 0.03152398, 0.05584872],
                    ],
                    [
                        [0.03152398, 0.11521591, 0.17348403],
                        [0.07065588, 0.30930897, 0.47804137],
                        [0.03152398, 0.15786578, 0.30930897],
                    ],
                    [
                        [0.00732843, 0.03152398, 0.05584872],
                        [0.03152398, 0.15786578, 0.30930897],
                        [0.05584872, 0.30930897, 0.77460594],
                    ],
                ],
            ]
        )
        assert allclose(pyscf_ao_2e_int.array, expected_ao_2e_int)

    def test_spatial_mo_2e_int(self) -> None:
        pyscf_spatial_mo_2e_int = PySCFSpatialMO2eInt(self.gto_mol, self.mo_coeff)
        expected_spatial_mo_2e_int = array(
            [
                [
                    [
                        [0.55936827, -0.0, 0.08803043],
                        [-0.0, 0.15477155, 0.0],
                        [0.08803043, 0.0, 0.13563321],
                    ],
                    [
                        [-0.0, 0.15477155, 0.0],
                        [0.48660662, -0.0, -0.0379191],
                        [0.0, -0.14084555, -0.0],
                    ],
                    [
                        [0.08803043, 0.0, 0.13563321],
                        [0.0, -0.14084555, -0.0],
                        [0.57575727, -0.0, 0.08987634],
                    ],
                ],
                [
                    [
                        [-0.0, 0.48660662, 0.0],
                        [0.15477155, -0.0, -0.14084555],
                        [0.0, -0.0379191, -0.0],
                    ],
                    [
                        [0.15477155, -0.0, -0.14084555],
                        [-0.0, 0.53466412, 0.0],
                        [-0.14084555, -0.0, 0.15039551],
                    ],
                    [
                        [0.0, -0.0379191, -0.0],
                        [-0.14084555, -0.0, 0.15039551],
                        [-0.0, 0.5123315, 0.0],
                    ],
                ],
                [
                    [
                        [0.08803043, 0.0, 0.57575727],
                        [0.0, -0.14084555, -0.0],
                        [0.13563321, -0.0, 0.08987634],
                    ],
                    [
                        [0.0, -0.14084555, -0.0],
                        [-0.0379191, 0.0, 0.5123315],
                        [-0.0, 0.15039551, 0.0],
                    ],
                    [
                        [0.13563321, -0.0, 0.08987634],
                        [-0.0, 0.15039551, 0.0],
                        [0.08987634, 0.0, 0.62546613],
                    ],
                ],
            ]
        )
        assert allclose(pyscf_spatial_mo_2e_int.array, expected_spatial_mo_2e_int)

    def test_active_space_spatial_integrals(self) -> None:
        asmo = ActiveSpaceMolecularOrbitals(self.pyscf_mol, cas(1, 1))
        nuc_energy = 1.3229430273
        pyscf_ao_1e_int = PySCFAO1eInt(self.gto_mol)
        pyscf_ao_2e_int = PySCFAO2eInt(self.gto_mol)

        electron_ints = PySCFAOeIntSet(
            mol=self.gto_mol,
            constant=nuc_energy,
            ao_1e_int=pyscf_ao_1e_int,
            ao_2e_int=pyscf_ao_2e_int,
        )
        spatial_as_mo_eint_set = get_active_space_spatial_integrals(asmo, electron_ints)
        as_mo_nuc_energy = spatial_as_mo_eint_set.const
        expected_nuc_energy = -1.18702694476004
        as_mo_1e_int = spatial_as_mo_eint_set.mo_1e_int.array
        expected_mo_1e_int = array([[-0.33696926]])
        as_mo_2e_int = spatial_as_mo_eint_set.mo_2e_int.array
        expected_mo_2e_int = array([[[[0.53466412]]]])

        assert isclose(as_mo_nuc_energy, expected_nuc_energy)
        assert allclose(as_mo_1e_int, expected_mo_1e_int)
        assert allclose(as_mo_2e_int, expected_mo_2e_int)


class TestSpinMOIntegralsFromMole:
    def test_get_spin_mo_integrals_from_mole_full_space(self) -> None:
        atoms = [["H", [0, 0, i]] for i in range(2)]
        gto_mol = gto.M(atom=atoms)
        mf = scf.RHF(gto_mol)
        mf.kernel()

        active_space, spin_mo_eint_set = get_spin_mo_integrals_from_mole(
            gto_mol, mf.mo_coeff
        )
        assert active_space.n_active_ele == 2
        assert active_space.n_active_orb == 2

        nuc_energy = spin_mo_eint_set.const
        spin_mo_1e_int_array = spin_mo_eint_set.mo_1e_int.array
        spin_mo_2e_int_array = spin_mo_eint_set.mo_2e_int.array

        expected_nuc_energy = 0.52917721092
        expected_spin_mo_1e_int = array(
            [
                [-1.11084418, 0.0, 0.0, 0.0],
                [0.0, -1.11084418, 0.0, 0.0],
                [0.0, 0.0, -0.589121, 0.0],
                [0.0, 0.0, 0.0, -0.589121],
            ]
        )

        expected_spin_mo_2e_int = array(
            [
                [
                    [
                        [0.6264025, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.19679058, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.6264025, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.19679058, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.19679058, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.62170676, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.19679058, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.62170676, 0.0, 0.0, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.6264025, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.19679058],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.6264025, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.19679058],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.19679058],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.62170676, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.19679058],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.62170676, 0.0, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.0, 0.62170676, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.19679058, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.62170676, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.19679058, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.19679058, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.65307075, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.19679058, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.65307075, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.0, 0.0, 0.62170676],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.19679058, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.62170676],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.19679058, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.19679058, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.65307075],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.19679058, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.65307075],
                    ],
                ],
            ]
        )

        assert isclose(nuc_energy, expected_nuc_energy)
        assert allclose(spin_mo_1e_int_array, expected_spin_mo_1e_int)
        assert allclose(spin_mo_2e_int_array, expected_spin_mo_2e_int)

    def test_get_spin_mo_integrals_from_mole_active_space(self) -> None:
        atoms = [["H", [0, 0, i]] for i in range(3)]
        gto_mol = gto.M(atom=atoms, spin=1)
        mf = scf.ROHF(gto_mol)
        mf.kernel()

        active_space, spin_mo_eint_set = get_spin_mo_integrals_from_mole(
            gto_mol, mf.mo_coeff, cas(1, 1)
        )

        nuc_energy = spin_mo_eint_set.const
        spin_mo_1e_int_array = spin_mo_eint_set.mo_1e_int.array
        spin_mo_2e_int_array = spin_mo_eint_set.mo_2e_int.array

        assert isinstance(active_space, ActiveSpace)
        assert active_space.n_active_ele == 1
        assert active_space.n_active_orb == 1

        expected_nuc_energy = -1.18702694476004
        expected_spin_mo_1e_int = array([[-0.33696926, 0.0], [0.0, -0.33696926]])
        expected_spin_mo_2e_int = array(
            [
                [[[0.53466412, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.53466412, 0.0]]],
                [[[0.0, 0.53466412], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.53466412]]],
            ]
        )

        assert isclose(nuc_energy, expected_nuc_energy)
        assert allclose(spin_mo_1e_int_array, expected_spin_mo_1e_int)
        assert allclose(spin_mo_2e_int_array, expected_spin_mo_2e_int)
