# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractproperty
from collections.abc import Collection, Sequence
from typing import Callable, Optional, Union

from openfermion.ops import FermionOperator, InteractionOperator, MajoranaOperator
from openfermion.transforms import bravyi_kitaev as of_bravyi_kitaev
from openfermion.transforms import bravyi_kitaev_tree, get_fermion_operator
from openfermion.transforms import jordan_wigner as of_jordan_wigner
from openfermion.transforms import reorder
from openfermion.transforms.opconversions import edit_hamiltonian_for_spin
from openfermion.transforms.opconversions.remove_symmetry_qubits import remove_indices
from openfermion.utils import up_then_down
from typing_extensions import TypeAlias

from quri_parts.chem.transforms import (
    BravyiKitaev,
    BravyiKitaevMapperFactory,
    FermionQubitMapperFactory,
    FermionQubitMapping,
    FermionQubitStateMapper,
    JordanWigner,
    JordanWignerMapperFactory,
    QubitFermionStateMapper,
    SymmetryConservingBravyiKitaev,
    SymmetryConservingBravyiKitaevMapperFactory,
)
from quri_parts.chem.utils.spin import occupation_state_sz
from quri_parts.core.operator import Operator, SinglePauli
from quri_parts.core.state import ComputationalBasisState
from quri_parts.core.utils.binary_field import BinaryArray, BinaryMatrix, inverse
from quri_parts.openfermion.operator import (
    has_particle_number_symmetry,
    operator_from_openfermion_op,
)

#: Interface for a function that maps a :class:`openfermion.ops.FermionOperator`,
#: :class:`openfermion.ops.InteractionOperator` or
#: :class:`openfermion.ops.MajoranaOperator` to
#: a :class:`Operator`.
OpenFermionQubitOperatorMapper: TypeAlias = Callable[
    [Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]],
    "Operator",
]


OpenFermionMappingMethods: TypeAlias = Union[
    "OpenFermionQubitMapping", "OpenFermionQubitMapperFactory"
]


def _inv_state_transformation_matrix(
    op_mapper: OpenFermionQubitOperatorMapper, n_spin_orbitals: int
) -> tuple[BinaryMatrix, Sequence[int]]:
    """Build a matrix that converts vector of fermion occupancy to binary
    representation on qubit.

    It is assumed that mapping method maps the occupancy to Z operator.
    For example, the matrix is (29) in http://arxiv.org/abs/1208.5986
    for the Bravyi-Kitaev method.
    """
    mat = []
    signs = []
    for number_op in (
        (1 - 2 * FermionOperator(f"{i}^ {i}")) for i in range(n_spin_orbitals)
    ):
        op = op_mapper(number_op)
        row = list(0 for _ in range(n_spin_orbitals))
        if len(op) > 1:
            raise ValueError(
                "This method is incompatible with a mapping that converts a number "
                "operator to a sum of multiple Pauli terms."
            )
        pauli_label, coef = next(iter(op.items()))

        assert coef == 1 or coef == -1
        signs.append(int(coef.real))

        for idx, pauli in pauli_label:
            if pauli != SinglePauli.Z:
                raise ValueError("The action must be Z.")
            row[idx] = 1
        mat.append(row)
    return BinaryMatrix(mat), signs


class OpenFermionQubitMapping(FermionQubitMapping, ABC):
    """Mapping object that holds the configuration of a state you want to
    transform or act operators onto. Operator mapper, state mapper and inverse
    state mapper are retrieved as properties.

    Args:
        n_spin_orbitals:
            The number of spin orbitals to be mapped to qubits.
        n_fermions:
            When specified, restrict the mapping to a subspace spanned by states
            containing the fixed number of Fermions. Some mappings require
            this argument (e.g. symmetry-conserving Bravyi-Kitaev transformation)
            while the others ignore it.
        sz:
            The spin along the z-axis of the state you want to transform. Some
            mappings require this argument (e.g. symmetry-conserving Bravyi-Kitaev
            transformation) while the others ignore it.
    """

    def __init__(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> None:
        if n_fermions is not None:
            assert n_fermions <= n_spin_orbitals, (
                "Number of n_fermions cannot be greater than n_spin_orbitals."
                f"Input {n_spin_orbitals=}, {n_fermions=}."
            )
        if sz is not None:
            assert float(
                2 * sz
            ).is_integer(), "sz should be either an integer or a half integer."

        self._n_spin_orbitals = n_spin_orbitals
        self._n_fermions = n_fermions
        self._sz = sz

        self._inv_trans_mat, self._signs = _inv_state_transformation_matrix(
            self.of_operator_mapper,
            self.n_spin_orbitals,
        )
        self._trans_mat = inverse(self._inv_trans_mat)

    @property
    def n_spin_orbitals(self) -> int:
        return self._n_spin_orbitals

    @property
    def n_fermions(self) -> Optional[int]:
        return self._n_fermions

    @property
    def sz(self) -> Optional[float]:
        return self._sz

    @abstractproperty
    def of_operator_mapper(self) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps an OpenFermion
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`.
        """
        ...

    @property
    def state_mapper(self) -> FermionQubitStateMapper:
        """Returns a function that maps occupied spin orbital indices to a
        computational basis state of qubits."""
        n_qubits = self.n_qubits
        n_spin_orbitals = self.n_spin_orbitals

        def mapper(
            occupied_indices: Collection[int],
        ) -> ComputationalBasisState:
            if len(set(occupied_indices)) != len(occupied_indices):
                raise ValueError(
                    "Repeated indices are not allowed in occupied_indeces."
                )

            if self.n_fermions is not None and len(occupied_indices) != self.n_fermions:
                raise ValueError(
                    f"Expected number of fermions to be {self.n_fermions}, "
                    f"but got {len(occupied_indices)}."
                )

            if self.sz is not None:
                state_sz = occupation_state_sz(occupied_indices)
                if self.sz != state_sz:
                    raise ValueError(
                        f"Expected sz of the state to be {self.sz}, "
                        f"but got sz={state_sz}."
                    )

            occ_list = [(i in occupied_indices) for i in range(n_spin_orbitals)]
            occ_list = [
                not b if self._signs[i] == -1 else b for i, b in enumerate(occ_list)
            ]
            occupancy_vector = BinaryArray(occ_list)
            qubit_vector = self._trans_mat @ occupancy_vector
            return ComputationalBasisState(
                n_qubits=n_qubits, bits=qubit_vector.binary & (2**n_qubits - 1)
            )

        return mapper

    @property
    def inv_state_mapper(self) -> QubitFermionStateMapper:
        """Returns a function that maps a computational basis state of qubits
        to the set of occupied spin orbital indices."""

        n_qubits = self.n_qubits
        n_spin_orbitals = self.n_spin_orbitals

        def mapper(state: ComputationalBasisState) -> Collection[int]:
            bits = state.bits
            bit_array = [(bits & 1 << index) >> index for index in range(n_qubits)]
            bit_array = self._augment_dropped_bits(
                bit_array, n_spin_orbitals, self.n_fermions, self.sz
            )
            qubit_vector = BinaryArray(bit_array)
            occupancy_vector = self._inv_trans_mat @ qubit_vector
            occupancy_set = [
                i
                for i, o in enumerate(occupancy_vector)
                if (o == 1 and self._signs[i] == 1) or (o == 0 and self._signs[i] == -1)
            ]
            return occupancy_set

        return mapper

    @staticmethod
    def _augment_dropped_bits(
        bit_array: list[int],
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> list[int]:
        """Returns a bit array which is augmented by adding qubits dropped by a
        :class:`FermionQubitMapping`."""
        return bit_array


class OpenFermionQubitMapperFactory(FermionQubitMapperFactory):
    """Mapping from Fermionic operators and states to :class:`Operator`s and
    states using OpenFermion."""

    _mapping_method: type[OpenFermionQubitMapping]

    def __call__(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> OpenFermionQubitMapping:
        """Returns a :class:`~OpenFermionQubitMapping` object."""
        return self._mapping_method(n_spin_orbitals, n_fermions, sz)

    def get_of_operator_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps an OpenFermion
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits.
            n_fermions:
                When specified, restrict the mapping to a subspace spanned by states
                containing the fixed number of Fermions. Some mappings require
                this argument (e.g. symmetry-conserving Bravyi-Kitaev transformation)
                while the others ignore it.
            sz:
                The spin along the z-axis of the state you want to transform. Some
                mappings require this argument (e.g. symmetry-conserving Bravyi-Kitaev
                transformation) while the others ignore it.
        """
        return self(n_spin_orbitals, n_fermions, sz).of_operator_mapper

    def get_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> FermionQubitStateMapper:
        """Returns a function that maps occupied spin orbital indices to a
        computational basis state of qubits.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits.
            n_fermions:
                When specified, restrict the mapping to a subspace spanned by states
                containing the fixed number of Fermions. Some mappings require
                this argument (e.g. symmetry-conserving Bravyi-Kitaev transformation)
                while the others ignore it.
            sz:
                The spin along the z-axis of the state you want to transform. Some
                mappings require this argument (e.g. symmetry-conserving Bravyi-Kitaev
                transformation) while the others ignore it.
        """
        return self(n_spin_orbitals, n_fermions, sz).state_mapper

    def get_inv_state_mapper(
        self,
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> QubitFermionStateMapper:
        """Returns a function that maps computational basis state of qubits to
        an occupied spin orbital indices.

        Args:
            n_spin_orbitals:
                The number of spin orbitals to be mapped to qubits.
            n_fermions:
                When specified, restrict the mapping to a subspace spanned by states
                containing the fixed number of Fermions. Some mappings require
                this argument (e.g. symmetry-conserving Bravyi-Kitaev transformation)
                while the others ignore it.
            sz:
                The spin along the z-axis of the state you want to transform. Some
                mappings require this argument (e.g. symmetry-conserving Bravyi-Kitaev
                transformation) while the others ignore it.
        """
        return self(n_spin_orbitals, n_fermions, sz).inv_state_mapper


class OpenFermionJordanWigner(JordanWigner, OpenFermionQubitMapping):
    """Jordan-Wigner transformation using OpenFermion.

    Args:
        n_spin_orbitals:
            The number of spin orbitals to be mapped to qubits.
        n_fermions:
            This is not used in the transformation thus being ignored.
        sz:
            This is not used in the transformation thus being ignored.
    """

    @property
    def of_operator_mapper(self) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps a
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`
        with Jordan-Wigner transformation.
        """

        def mapper(
            op: Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]
        ) -> "Operator":
            return operator_from_openfermion_op(of_jordan_wigner(op))

        return mapper


class OpenFermionJordanWignerFactory(
    JordanWignerMapperFactory, OpenFermionQubitMapperFactory
):
    """Jordan-Wigner transformation using OpenFermion."""

    _mapping_method = OpenFermionJordanWigner


jordan_wigner = OpenFermionJordanWignerFactory()
r"""An object that performs Jordan-Wigner mapping in various ways.

Example:
    You may create mappers out of `jordan_wigner`

    >>> operator_mapper = jordan_wigner.get_of_operator_mapper(8)
    >>> operator_mapper(FermionOperator("1^ 1"))
    (0.5+0j)*I + (-0.5+0j)*Z1

    >>> state_mapper = jordan_wigner.get_state_mapper(8)
    >>> state_mapper([0, 1])
    ComputationalBasisState(qubit_count=8, bits=0b11, phase=0π/2)

    >>> inv_state_mapper = jordan_wigner.get_inv_state_mapper(8)
    >>> inv_state_mapper(ComputationalBasisState(8, bits=0b11))
    [0, 1]

    You may create a mapping object with specified number of spin orbials.

    >>> jw_mapping = jordan_wigner(8)

    >>> operator_mapper = jw_mapping.of_operator_mapper
    >>> operator_mapper(FermionOperator("1^ 1"))
    (0.5+0j)*I + (-0.5+0j)*Z1

    >>> state_mapper = jw_mapping.state_mapper
    >>> state_mapper([0, 1])
    ComputationalBasisState(qubit_count=8, bits=0b11, phase=0π/2)

    >>> inv_state_mapper = jw_mapping.inv_state_mapper
    >>> inv_state_mapper(ComputationalBasisState(8, bits=0b11))
    [0, 1]
"""


class OpenFermionBravyiKitaev(BravyiKitaev, OpenFermionQubitMapping):
    """Bravyi-Kitaev transformation using OpenFermion.

    Args:
        n_spin_orbitals:
            The number of spin orbitals to be mapped to qubits.
        n_fermions:
            This is not used in the transformation thus being ignored.
        sz:
            This is not used in the transformation thus being ignored.
    """

    @property
    def of_operator_mapper(self) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps a
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`
        with Bravyi-Kitaev transformation.
        """

        def mapper(
            op: Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]
        ) -> "Operator":
            return operator_from_openfermion_op(
                of_bravyi_kitaev(op, n_qubits=self.n_spin_orbitals)
            )

        return mapper


class OpenFermionBravyiKitaevFactory(
    BravyiKitaevMapperFactory, OpenFermionQubitMapperFactory
):
    """Bravyi-Kitaev transformation using OpenFermion."""

    _mapping_method = OpenFermionBravyiKitaev


bravyi_kitaev = OpenFermionBravyiKitaevFactory()
r"""An object that performs Bravyi-Kitaev mapping in various ways.

Example:
    You may create mappers out of `bravyi_kitaev`

    >>> operator_mapper = bravyi_kitaev.get_of_operator_mapper(8)
    >>> print(operator_mapper(FermionOperator("1^ 1")))
    (0.5+0j)*I + (-0.5+0j)*Z0 Z1

    >>> state_mapper = bravyi_kitaev.get_state_mapper(8)
    >>> state_mapper([0, 1])
    ComputationalBasisState(qubit_count=8, bits=0b1, phase=0π/2)

    >>> inv_state_mapper = bravyi_kitaev.get_inv_state_mapper(8)
    >>> inv_state_mapper(ComputationalBasisState(8, bits=0b10111))
    [0, 2, 4, 5]

    You may create a mapping object with specified number of spin orbials.

    >>> bk_mapping = bravyi_kitaev(8)

    >>> operator_mapper = bk_mapping.of_operator_mapper
    >>> print(operator_mapper(FermionOperator("1^ 1")))
    (0.5+0j)*I + (-0.5+0j)*Z0 Z1

    >>> state_mapper = bk_mapping.state_mapper
    >>> state_mapper([0, 1])
    ComputationalBasisState(qubit_count=8, bits=0b1, phase=0π/2)

    >>> inv_state_mapper = bk_mapping.inv_state_mapper
    >>> inv_state_mapper(ComputationalBasisState(8, bits=0b10111))
    [0, 2, 4, 5]
"""


def _get_scbk_parity_factor(n_fermions: int, sz: float) -> tuple[int, int]:
    n_spin_ups = int(0.5 * (n_fermions + 2 * sz))
    return (-1) ** n_spin_ups, (-1) ** n_fermions


class OpenFermionSymmetryConservingBravyiKitaev(
    SymmetryConservingBravyiKitaev, OpenFermionQubitMapping
):
    """Symmetry-conserving Bravyi-Kitaev transformation described in
    arXiv:1701.08213, using OpenFermion.

    Note that in this mapping the spin orbital indices are first
    reordered to all spin-up orbitals, then all spin-down orbitals.
    Bravyi-Kitaev transoformation is applied after the reordering and
    then two qubits are dropped using conservation of particle number
    and spin.

    Any operators which don't have particle number and spin symmetry are
    converted to `Operator()`, whose expectation value is zero for all
    states.

    Args:
        n_spin_orbitals:
            The number of spin orbitals to be mapped to qubits.
        n_fermions:
            The number of fermions the state should contain.
        sz:
            Spin along the z-axis of the state. Currently, only sz = 0
            and 0.5 are supported.
    """

    @property
    def of_operator_mapper(self) -> OpenFermionQubitOperatorMapper:
        """Returns a function that maps a
        :class:`~openfermion.ops.FermionOperator`,
        :class:`~openfermion.ops.InteractionOperator` or
        :class:`~openfermion.ops.MajoranaOperator`
        to a :class:`openfermion.ops.QubitOperator`
        with symmetry-conserving Bravyi-Kitaev transformation.

        Both the arguments (``n_spin_orbitals`` and ``n_fermions``) are required.
        """

        if self.n_fermions is None:
            raise ValueError("n_fermions is required.")
        if self.sz is None:
            raise ValueError("sz is required.")

        n_spin_orbitals = self.n_spin_orbitals
        n_fermions = self.n_fermions
        sz = self.sz

        def mapper(
            op: Union["FermionOperator", "InteractionOperator", "MajoranaOperator"]
        ) -> "Operator":
            if not isinstance(op, FermionOperator):
                op = get_fermion_operator(op)

            symmetry_op = FermionOperator()
            for op_tuple, coeff in op.terms.items():
                single_op = FermionOperator(op_tuple, coeff)
                if has_particle_number_symmetry(single_op, check_spin_symmetry=True):
                    symmetry_op += single_op

            (
                mid_parity_factor,
                last_parity_factor,
            ) = _get_scbk_parity_factor(n_fermions, sz)

            transformed_op = bravyi_kitaev_tree(
                reorder(symmetry_op, up_then_down, num_modes=n_spin_orbitals),
                n_qubits=n_spin_orbitals,
            )
            operator_tappered = edit_hamiltonian_for_spin(
                transformed_op,
                spin_orbital=n_spin_orbitals,
                orbital_parity=last_parity_factor,
            )
            operator_tappered = edit_hamiltonian_for_spin(
                operator_tappered,
                spin_orbital=n_spin_orbitals // 2,
                orbital_parity=mid_parity_factor,
            )
            operator_tappered = remove_indices(
                operator_tappered, [n_spin_orbitals // 2, n_spin_orbitals]
            )

            return operator_from_openfermion_op(operator_tappered)

        return mapper

    @staticmethod
    def _augment_dropped_bits(
        bit_array: list[int],
        n_spin_orbitals: int,
        n_fermions: Optional[int] = None,
        sz: Optional[float] = None,
    ) -> list[int]:
        # Add two qubits dropped by the fermion-to-qubit mapping.
        return bit_array + [0, 0]


class OpenFermionSymmetryConservingBravyiKitaevFactory(
    SymmetryConservingBravyiKitaevMapperFactory, OpenFermionQubitMapperFactory
):
    """Symmetry-conserving Bravyi-Kitaev transformation described in
    arXiv:1701.08213, using OpenFermion.

    Note that in this mapping the spin orbital indices are first
    reordered to all spin-up orbitals, then all spin-down orbitals.
    Bravyi-Kitaev transoformation is applied after the reordering and
    then two qubits are dropped using conservation of particle number
    and spin.

    Any operators which don't have particle number and spin symmetry are
    converted to `Operator()`, whose expectation value is zero for all
    states.
    """

    _mapping_method = OpenFermionSymmetryConservingBravyiKitaev


symmetry_conserving_bravyi_kitaev = OpenFermionSymmetryConservingBravyiKitaevFactory()
r"""An object that performs spin-symmetric Bravyi-Kitaev mapping in various
ways.

Example:
    You may create mappers out of `symmetry_conserving_bravyi_kitaev`

    >>> scbk = symmetry_conserving_bravyi_kitaev

    >>> operator_mapper = scbk.get_of_operator_mapper(8, 4, 0)
    >>> print(operator_mapper(FermionOperator("1^ 1")))
    0.5*I + -0.5*Z3

    >>> state_mapper = scbk.get_state_mapper(8, 4, 0)
    >>> state_mapper([0, 1, 2, 3])
    ComputationalBasisState(qubit_count=6, bits=0b1001, phase=0π/2)

    >>> inv_state_mapper = scbk.get_inv_state_mapper(8, 4, 0)
    >>> inv_state_mapper(ComputationalBasisState(qubit_count=6, bits=0b1001))
    [0, 1, 2, 3]

    You may create a mapping object with specified number of spin orbials,
    number of electrons and sz of the state.

    >>> scbk_mapping = scbk(8, 4, 0)

    >>> operator_mapper = scbk_mapping.of_operator_mapper
    >>> print(operator_mapper(FermionOperator("1^ 1")))
    0.5*I + -0.5*Z3

    >>> state_mapper = scbk_mapping.state_mapper
    >>> state_mapper([0, 1, 2, 3])
    ComputationalBasisState(qubit_count=6, bits=0b1001, phase=0π/2)

    >>> inv_state_mapper = scbk_mapping.inv_state_mapper
    >>> inv_state_mapper(ComputationalBasisState(qubit_count=6, bits=0b1001))
    [0, 1, 2, 3]
"""
