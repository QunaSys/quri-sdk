# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

from openfermion import FermionOperator, InteractionOperator, MajoranaOperator

from quri_parts.chem.mol import ActiveSpace, SpinMOeIntSet
from quri_parts.core.operator import Operator
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapperFactory,
    OpenFermionQubitMapping,
    jordan_wigner,
)


def get_fermionic_hamiltonian(
    spin_mo_eint_set: SpinMOeIntSet,
) -> InteractionOperator:
    """Construct the molecular hamiltonian from the spin MO electron
    integrals."""
    nuc_energy = spin_mo_eint_set.const
    mo_1e_int = spin_mo_eint_set.mo_1e_int.array
    mo_2e_int = spin_mo_eint_set.mo_2e_int.array / 2
    return InteractionOperator(nuc_energy, mo_1e_int, mo_2e_int)


def operator_from_of_fermionic_op(
    fermionic_hamiltonian: Union[
        FermionOperator, InteractionOperator, MajoranaOperator
    ],
    active_space: ActiveSpace,
    sz: Optional[float] = None,
    fermion_qubit_mapping: OpenFermionQubitMapperFactory = jordan_wigner,
) -> tuple[Operator, OpenFermionQubitMapping]:
    """Converts the fermionic hamiltonian into qubit hamiltonian with a given
    mapping method, and returns the operator mapper along with the state
    mapper."""

    n_spin_orbitals = 2 * active_space.n_active_orb
    n_electrons = active_space.n_active_ele
    mapping = fermion_qubit_mapping(n_spin_orbitals, n_electrons, sz)
    operator_mapper = mapping.of_operator_mapper
    return operator_mapper(fermionic_hamiltonian), mapping


def get_qubit_mapped_hamiltonian(
    active_space: ActiveSpace,
    spin_mo_eint_set: SpinMOeIntSet,
    sz: Optional[float] = None,
    fermion_qubit_mapping: OpenFermionQubitMapperFactory = jordan_wigner,
) -> tuple[Operator, OpenFermionQubitMapping]:
    """Computes the qubit hamiltonian and returns the operator mapper along
    with the state mapper."""
    fermionic_hamiltonian = get_fermionic_hamiltonian(spin_mo_eint_set)
    return operator_from_of_fermionic_op(
        fermionic_hamiltonian, active_space, sz, fermion_qubit_mapping
    )
