# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .active_space import (
    convert_to_spin_orbital_indices,
    get_core_and_active_orbital_indices,
)
from .models import (
    ActiveSpace,
    ActiveSpaceMolecularOrbitals,
    AO1eInt,
    AO2eInt,
    AOeIntSet,
    MolecularOrbitals,
    SpatialMO1eInt,
    SpatialMO1eIntArray,
    SpatialMO2eInt,
    SpatialMO2eIntArray,
    SpatialMOeIntSet,
    SpinMO1eInt,
    SpinMO1eIntArray,
    SpinMO2eInt,
    SpinMO2eIntArray,
    SpinMOeIntSet,
    cas,
)
from .non_relativistic_models import (
    AO1eIntArray,
    AO2eIntArray,
    AOeIntArraySet,
    get_active_space_spatial_integrals_from_ao_eint,
    get_active_space_spatial_integrals_from_mo_eint,
    get_active_space_spin_integrals_from_ao_eint,
    get_active_space_spin_integrals_from_mo_eint,
    get_effective_active_space_1e_integrals,
    get_effective_active_space_2e_integrals,
    get_effective_active_space_core_energy,
    spatial_mo_1e_int_to_spin_mo_1e_int,
    spatial_mo_2e_int_to_spin_mo_2e_int,
    spatial_mo_eint_set_to_spin_mo_eint_set,
    to_spin_orbital_integrals,
)

__all__ = [
    "ActiveSpace",
    "cas",
    "ActiveSpaceMolecularOrbitals",
    "AO1eInt",
    "AO2eInt",
    "AOeIntSet",
    "AO1eIntArray",
    "AO2eIntArray",
    "AOeIntArraySet",
    "MolecularOrbitals",
    "SpatialMO1eInt",
    "SpatialMO1eIntArray",
    "SpatialMO2eInt",
    "SpatialMO2eIntArray",
    "SpatialMOeIntSet",
    "SpinMO1eInt",
    "SpinMO2eInt",
    "SpinMO1eIntArray",
    "SpinMO2eIntArray",
    "SpinMOeIntSet",
    "convert_to_spin_orbital_indices",
    "get_active_space_spin_integrals_from_ao_eint",
    "get_active_space_spatial_integrals_from_ao_eint",
    "get_active_space_spin_integrals_from_mo_eint",
    "get_active_space_spatial_integrals_from_mo_eint",
    "get_core_and_active_orbital_indices",
    "get_effective_active_space_core_energy",
    "get_effective_active_space_1e_integrals",
    "get_effective_active_space_2e_integrals",
    "spatial_mo_eint_set_to_spin_mo_eint_set",
    "to_spin_orbital_integrals",
    "spatial_mo_1e_int_to_spin_mo_1e_int",
    "spatial_mo_2e_int_to_spin_mo_2e_int",
]
