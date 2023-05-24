# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Optional, Protocol, Sequence

import numpy as np
import numpy.typing as npt  # noqa: F401

from quri_parts.chem.mol import get_core_and_active_orbital_indices


class OrbitalType(Enum):
    """Type of orbital."""

    #: core orbital.
    CORE = auto()
    #: active orbital.
    ACTIVE = auto()
    #: virtual orbital.
    VIRTUAL = auto()


@dataclass(frozen=True)
class ActiveSpace:
    """An immutable (frozen) dataclass representing a active space."""

    #: number of active electrons.
    n_active_ele: int
    #: number of active orbitals.
    n_active_orb: int
    #: sequence of spatial orbital indices of the active orbitals.
    active_orbs_indices: Optional[Sequence[int]] = None


def cas(
    n_active_ele: int,
    n_active_orb: int,
    active_orbs_indices: Optional[Sequence[int]] = None,
) -> ActiveSpace:
    """Convenient function for constructing the active space."""
    return ActiveSpace(n_active_ele, n_active_orb, active_orbs_indices)


class MolecularOrbitals(Protocol):
    """Interface protocol for a data of the molecule."""

    @abstractproperty
    def n_electron(self) -> int:
        """Returns the number of electrons."""
        ...

    @abstractproperty
    def spin(self) -> int:
        """Returns the total spin of the electrons.

        Note:
            We follow the quantum chemistry convention where:

            .. math::
                N_{\\alpha} - N_{\\beta} = 2S

            and spin = 2S.
        """
        ...

    @abstractproperty
    def n_spatial_orb(self) -> int:
        """Returns the number of spatial orbitals."""
        ...

    @abstractproperty
    def mo_coeff(self) -> "npt.NDArray[np.complex128]":
        """Returns molecular orbital coefficients."""
        ...


class ActiveSpaceMolecularOrbitals(MolecularOrbitals):
    """Represents a data of the active space for the molecule.

    Args:
        mo: :class:`MolecularOrbitals` for the molecule.
        active_space: :class:`ActiveSpace` for the molecule.
    """

    def __init__(
        self,
        mo: MolecularOrbitals,
        active_space: ActiveSpace,
    ):
        self._mo = mo
        self._active_space = active_space
        self._check_active_space_consistency()

    @property
    def n_electron(self) -> int:
        """Returns a number of electrons."""
        return self._mo.n_electron

    @property
    def spin(self) -> int:
        """Returns the total spin of the electrons."""
        return self._mo.spin

    @property
    def n_active_ele(self) -> int:
        """Returns the number of active electrons."""
        return self._active_space.n_active_ele

    @property
    def n_core_ele(self) -> int:
        """Returns the number of core electrons."""
        n_core_electron = self.n_electron - self.n_active_ele
        return n_core_electron

    @property
    def n_ele_alpha(self) -> int:
        """Return the number of spin up electrons."""
        return self.n_active_ele - self.n_ele_beta

    @property
    def n_ele_beta(self) -> int:
        """Return the number of spin down electrons."""
        return (self.n_active_ele - self.spin) // 2

    @property
    def n_spatial_orb(self) -> int:
        """Returns the number of spatial orbitals."""
        return self._mo.n_spatial_orb

    @property
    def n_active_orb(self) -> int:
        """Returns the number of active orbitals."""
        return self._active_space.n_active_orb

    @property
    def n_core_orb(self) -> int:
        """Returns the number of core orbitals."""
        return self.n_core_ele // 2

    @property
    def n_vir_orb(self) -> int:
        """Returns the number of virtual orbitals."""
        return self._mo.n_spatial_orb - self.n_active_orb - self.n_core_orb

    @property
    def mo_coeff(self) -> "npt.NDArray[np.complex128]":
        """Returns molecular orbital coefficients."""
        return self._mo.mo_coeff

    def get_core_and_active_orb(self) -> tuple[Sequence[int], Sequence[int]]:
        """Returns a set of core and active orbitals.

        The output orbitals are represented as the spatial orbital.
        """
        n_active_ele = self._active_space.n_active_ele
        n_active_orb = self._active_space.n_active_orb
        active_orbs_indices = self._active_space.active_orbs_indices
        core_orb_list, active_orb_list = get_core_and_active_orbital_indices(
            n_active_ele,
            n_active_orb,
            self._mo.n_electron,
            active_orbs_indices,
        )
        return core_orb_list, active_orb_list

    def orb_type(self, mo_index: int) -> OrbitalType:
        """Returns a type of the given orbital index.

        Args:
            mo_index: Orbital index.
        """
        core_orb_list, active_orb_list = self.get_core_and_active_orb()
        if mo_index in core_orb_list:
            return OrbitalType.CORE
        elif mo_index in active_orb_list:
            return OrbitalType.ACTIVE
        else:
            return OrbitalType.VIRTUAL

    def _check_active_space_consistency(self) -> None:
        """Consistency check of the active space configuration."""
        assert self.n_core_ele % 2 == 0, ValueError(
            "The number of electrons in core must be even."
            " Please set the active electron to an {} number".format(
                {0: "even", 1: "odd"}[self.n_electron % 2]
            )
        )
        assert self.n_core_ele >= 0, ValueError(
            f"Number of core electrons should be a positive integer or zero.\n"
            f" n_core_ele = {self.n_core_ele}.\n"
            f" Possible fix: n_active_ele should be less than"
            f" total number of electrons: {self.n_electron}."
        )
        assert self.n_vir_orb >= 0, ValueError(
            f"Number of virtual orbitals should be a positive integer or zero.\n"
            f" n_vir = {self.n_vir_orb}.\n"
            f" Possible fix: (n_active_orb - n_active_ele//2) should not"
            f" exceed {self.n_spatial_orb - self.n_electron//2}."
        )
        assert self.n_ele_alpha >= 0, ValueError(
            f"Number of spin up electrons should be a positive integer or zero.\n"
            f" n_ele_alpha = {self.n_ele_alpha}"
            f" Possible_fix: - n_active_ele should not"
            f" be less then the value of spin: {self.spin}."
        )
        assert self.n_ele_beta >= 0, ValueError(
            f"Number of spin down electrons should be a positive integer or zero.\n"
            f" n_ele_beta = {self.n_ele_beta}.\n"
            f" Possible_fix: n_active_ele should not"
            f" exceed the value of spin: {self.spin}."
        )
        assert self.n_ele_alpha <= self.n_active_orb, ValueError(
            f"Number of spin up electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_alpha = {self.n_ele_alpha},\n"
            f" n_active_orb = {self.n_active_orb}.\n"
            f" Possible fix: [(n_active_ele + spin)//2] should be"
            f" less than n_active_orb: {self.n_active_orb}"
        )
        assert self.n_ele_beta <= self.n_active_orb, ValueError(
            f"Number of spin down electrons should not exceed the number of active orbitals.\n"  # noqa: E501
            f" n_ele_beta = {self.n_ele_beta},\n"
            f" n_active_orb = {self.n_active_orb}\n"
            f" Possible fix: [(n_active_ele - spin)//2] should be"
            f" less than n_active_orb: {self.n_active_orb}"
        )

    def __str__(self) -> str:
        info_dict = {
            "n_electron": self.n_electron,
            "n_active_ele": self.n_active_ele,
            "n_core_ele": self.n_core_ele,
            "n_ele_alpha": self.n_ele_alpha,
            "n_ele_beta": self.n_ele_beta,
            "n_spatial_orb": self.n_spatial_orb,
            "n_active_orb": self.n_active_orb,
            "n_core_orb": self.n_core_orb,
            "n_vir_orb": self.n_vir_orb,
        }

        info_str = "\n".join(f"{key}: {str(value)}" for key, value in info_dict.items())

        return info_str


class AO1eInt(Protocol):
    """Interface protocol for an atomic orbital one-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...

    @abstractmethod
    def to_mo1int(self, mo_coeff: "npt.NDArray[np.complex128]") -> "SpinMO1eInt":
        """This method converts an atomic orbital one-electron integral into
        the molecular orbital one-electron integral.

        Args:
            mo_coeff: molecular orbital coefficients.
        """
        ...


class AO2eInt(Protocol):
    """Interface protocol for an atomic orbital two-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...

    @abstractmethod
    def to_mo2int(self, mo_coeff: "npt.NDArray[np.complex128]") -> "SpinMO2eInt":
        """This method converts an atomic orbital two-electron integral into
        the molecular orbital two-electron integral.

        Args:
            mo_coeff: molecular orbital coefficients.
        """
        ...


class AOeIntSet(Protocol):
    """AOeIntSet holds a constant and the atomic orbital electron integrals."""

    #: constant.
    constant: float
    #: atomic orbital one-electron integral :class:`AO1eInt`.
    ao_1e_int: AO1eInt
    #: atomic orbital two-electron integral :class:`AO2eInt`.
    ao_2e_int: AO2eInt

    def to_full_space_mo_int(
        self,
        mo: MolecularOrbitals,
    ) -> "SpinMOeIntSet":
        """Compute the full space spin mo integral."""
        ...

    def to_active_space_mo_int(
        self,
        active_space_mo: ActiveSpaceMolecularOrbitals,
    ) -> "SpinMOeIntSet":
        """Compute the active space spin mo integral."""
        ...


class SpatialMO1eInt(Protocol):
    """Interface protocol for a molecular orbital one-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...


class SpatialMO2eInt(Protocol):
    """Interface protocol for a molecular orbital two-electron integral."""

    @abstractproperty
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        ...


class SpatialMO1eIntArray(SpatialMO1eInt):
    """A class of a molecular orbital one-electron integral.

    This interface has an integral as numpy ndarray.
    """

    def __init__(self, array: "npt.NDArray[np.complex128]"):
        self._array = array

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        return self._array


class SpatialMO2eIntArray(SpatialMO2eInt):
    """A class of a molecular orbital two-electron integral.

    This interface has an integral as numpy ndarray.
    """

    def __init__(self, array: "npt.NDArray[np.complex128]"):
        self._array = array

    @property
    def array(self) -> "npt.NDArray[np.complex128]":
        """Returns integral as numpy ndarray."""
        return self._array


class SpatialMOeIntSet(NamedTuple):
    """SpatialMOeIntSet holds a constant and a set of molecular orbital
    electron integral."""

    #: constant.
    const: float
    #: molecular orbital one-electron integral.
    mo_1e_int: SpatialMO1eInt
    #: molecular orbital two-electron integral.
    mo_2e_int: SpatialMO2eInt


class SpinMO1eInt(Protocol):
    """Interface protocol for a molecular spin orbital one-electron
    integral."""

    @abstractproperty
    def array(self) -> npt.NDArray[np.complex128]:
        """Returns integral as numpy ndarray."""
        ...


class SpinMO2eInt(Protocol):
    """Interface protocol for a molecular spin orbital two-electron
    integral."""

    @abstractproperty
    def array(self) -> npt.NDArray[np.complex128]:
        """Returns integral as numpy ndarray."""
        ...


class SpinMO1eIntArray(SpinMO1eInt):
    """Stores the array of the 1-electron integrals."""

    def __init__(self, array: npt.NDArray[np.complex128]) -> None:
        self._array = array

    @property
    def array(self) -> npt.NDArray[np.complex128]:
        """Returns integral as numpy ndarray."""
        return self._array


class SpinMO2eIntArray(SpinMO2eInt):
    """Stores the array of the 2-electron integrals."""

    def __init__(self, array: npt.NDArray[np.complex128]) -> None:
        self._array = array

    @property
    def array(self) -> npt.NDArray[np.complex128]:
        """Returns integral as numpy ndarray."""
        return self._array


class SpinMOeIntSet(NamedTuple):
    """SpinMOeIntSet holds a constant and a set of molecular orbital electron
    integral."""

    #: constant.
    const: float
    #: spin molecular orbital one-electron integral.
    mo_1e_int: SpinMO1eInt
    #: spin molecular orbital two-electron integral.
    mo_2e_int: SpinMO2eInt
