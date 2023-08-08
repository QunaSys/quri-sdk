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

import numpy as np
import numpy.typing as npt  # noqa: F401
import pytest

from quri_parts.chem.mol import (
    ActiveSpace,
    ActiveSpaceMolecularOrbitals,
    MolecularOrbitals,
)


@dataclass
class MolecularOrbitalsInfo(MolecularOrbitals):
    _n_electron: int
    _n_spatial_orb: int
    _spin: int = 0
    charge: int = 0

    @property
    def n_electron(self) -> int:
        return self._n_electron

    @property
    def spin(self) -> int:
        return self._spin

    @property
    def n_spatial_orb(self) -> int:
        return self._n_spatial_orb

    @property
    def mo_coeff(self) -> npt.NDArray[np.complex128]:
        return np.zeros((1,), dtype=np.complex128)


h2o = MolecularOrbitalsInfo(_n_electron=10, _n_spatial_orb=7)
charged_h2o = MolecularOrbitalsInfo(_n_electron=9, _n_spatial_orb=7, _spin=1, charge=1)
spinning_h2o = MolecularOrbitalsInfo(_n_electron=10, _n_spatial_orb=24, _spin=6)
neg_spinning_h2o = MolecularOrbitalsInfo(_n_electron=10, _n_spatial_orb=24, _spin=-6)


def test_odd_core_ele() -> None:
    with pytest.raises(
        AssertionError,
        match=(
            "The number of electrons in core must be even."
            " Please set the active electron to an even number"
        ),
    ):
        ActiveSpaceMolecularOrbitals(h2o, ActiveSpace(n_active_ele=3, n_active_orb=2))


def test_neg_core_ele() -> None:
    n_active_ele = 100
    n_active_orb = 2
    n_electron = h2o.n_electron
    n_core_ele = n_electron - n_active_ele
    with pytest.raises(
        AssertionError,
        match=(
            f"Number of core electrons should be a positive integer or zero.\n"
            f" n_core_ele = {n_core_ele}.\n"
            f" Possible fix: n_active_ele should be less than"
            f" total number of electrons: {n_electron}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            h2o, ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb)
        )


def test_neg_vir_orb() -> None:
    n_active_ele = 6
    n_active_orb = 6
    n_electron = h2o.n_electron
    n_spatial_orb = h2o.n_spatial_orb
    n_core_ele = n_electron - n_active_ele
    n_core_orb = n_core_ele // 2
    n_vir_orb = n_spatial_orb - n_active_orb - n_core_orb
    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of virtual orbitals should be a positive integer or zero.\n"
            f" n_vir = {n_vir_orb}.\n"
            f" Possible fix: (n_active_orb - n_active_ele//2) should not"
            f" exceed {n_spatial_orb - n_electron//2}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            h2o, ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb)
        )


def test_neg_vir_orb_2() -> None:
    n_active_ele = 7
    n_active_orb = 7
    n_electron = charged_h2o.n_electron
    n_spatial_orb = charged_h2o.n_spatial_orb
    n_core_ele = n_electron - n_active_ele
    n_core_orb = n_core_ele // 2
    n_vir_orb = n_spatial_orb - n_active_orb - n_core_orb

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of virtual orbitals should be a positive integer or zero.\n"
            f" n_vir = {n_vir_orb}.\n"
            f" Possible fix: (n_active_orb - n_active_ele//2) should not"
            f" exceed {n_spatial_orb - n_electron//2}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            charged_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_neg_alpha_elec() -> None:
    n_active_ele = 4
    n_active_orb = 4
    spin = neg_spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2
    n_ele_alpha = n_active_ele - n_ele_beta

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin up electrons should be a positive integer or zero.\n"
            f" n_ele_alpha = {n_ele_alpha}"
            f" Possible_fix: - n_active_ele should not"
            f" be less then the value of spin: {spin}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            neg_spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_neg_beta_elec() -> None:
    n_active_ele = 4
    n_active_orb = 4
    spin = spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin down electrons should be a positive integer or zero.\n"
            f" n_ele_beta = {n_ele_beta}.\n"
            f" Possible_fix: n_active_ele should not"
            f" exceed the value of spin: {spin}."
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_large_alpha_ele() -> None:
    n_active_ele = 8
    n_active_orb = 2
    spin = spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2
    n_ele_alpha = n_active_ele - n_ele_beta

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin up electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_alpha = {n_ele_alpha},\n"
            f" n_active_orb = {n_active_orb}.\n"
            f" Possible fix: [(n_active_ele + spin)//2] should be"
            f" less than n_active_orb: {n_active_orb}"
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )


def test_large_beta_ele() -> None:
    n_active_ele = 8
    n_active_orb = 2
    spin = neg_spinning_h2o.spin
    n_ele_beta = (n_active_ele - spin) // 2

    import re

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Number of spin down electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_beta = {n_ele_beta},\n"
            f" n_active_orb = {n_active_orb}\n"
            f" Possible fix: [(n_active_ele - spin)//2] should be"
            f" less than n_active_orb: {n_active_orb}"
        ),
    ):
        ActiveSpaceMolecularOrbitals(
            neg_spinning_h2o,
            ActiveSpace(n_active_ele=n_active_ele, n_active_orb=n_active_orb),
        )
