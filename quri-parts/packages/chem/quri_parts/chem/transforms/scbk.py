# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import numpy

from quri_parts.chem.transforms.fermion import FermionLabel
from quri_parts.chem.transforms.fermion_operator import FermionOperator
from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    PauliLabel,
    SinglePauli,
    pauli_label,
)


def has_particle_number_symmetry(
    f_op: "FermionOperator", check_spin_symmetry: bool = False
) -> bool:
    """Query whether operator conserves particle number and (optionally) spin
    symmetry.

    Args:
        f_op: fermionic operator (FermionOperator)
        check_spin_symmetry: boolean value
    Returns:
        bool
    """

    for term, coef in f_op.items():
        spin = 0
        particles = 0
        for operator in term:
            particles += (-1) ** (operator[1] + 1)  # add 1 if create, else -1
            spin += (-1) ** (operator[0] + operator[1])
        if particles:
            return False
        elif spin and check_spin_symmetry:
            return False
    return True


def up_then_down(mode_idx: int, num_modes: int) -> int:
    """Up then down reordering, given the operator has the default even-odd
    ordering. Otherwise this function will reorder indices where all even
    indices now come before odd indices.

     Example:
         0,1,2,3,4,5 -> 0,2,4,1,3,5

    The function takes in the index of the mode that will be relabeled and
    the total number modes.

    Args:
        mode_idx (int): the mode index that is being reordered
        num_modes (int): the total number of modes of the operator.

    Returns (int): reordered index of the mode.
    """
    halfway = int(numpy.ceil(num_modes / 2.0))

    if mode_idx % 2 == 0:
        return mode_idx // 2

    return mode_idx // 2 + halfway


def reorder(
    operator: FermionOperator,
    order_function: Callable[[int, int], int],
    num_modes: int,
    reverse: bool = False,
) -> FermionOperator:
    """Changes the ladder operator order of the Hamiltonian based on the
    provided order_function per mode index.

    Args:
        operator (SymbolicOperator): the operator that will be reordered. must
            be a SymbolicOperator or any type of operator that inherits from
            SymbolicOperator.
        order_function (func): a function per mode that is used to map the
            indexing. must have arguments mode index and num_modes.
        num_modes (int): default None. User can provide the number of modes
            assumed for the system. if None, the number of modes will be
            calculated based on the Operator.
        reverse (bool): default False. if set to True, the mode mapping is
            reversed. reverse = True will not revert back to original if
            num_modes calculated differs from original and reverted.

    Note: Every order function must take in a mode_idx and num_modes.
    """

    mode_map = {
        mode_idx: order_function(mode_idx, num_modes) for mode_idx in range(num_modes)
    }

    if reverse:
        mode_map = {val: key for key, val in mode_map.items()}

    rotated_hamiltonian = FermionOperator()
    for term, value in operator.items():
        new_term = tuple([(mode_map[op[0]], op[1]) for op in term])
        rotated_hamiltonian += FermionOperator({FermionLabel(new_term): value})
    return rotated_hamiltonian


def edit_hamiltonian_for_spin(
    qubit_hamiltonian: Operator, spin_orbital: int, orbital_parity: int
) -> Operator:
    """Removes the Z terms acting on the orbital from the Hamiltonian."""
    new_qubit_dict: dict[PauliLabel, complex] = {}
    for term, coefficient in qubit_hamiltonian.items():
        # If Z acts on the specified orbital, precompute its effect and
        # remove it from the Hamiltonian.
        if (spin_orbital - 1, SinglePauli.Z) in term:
            new_coefficient = coefficient * orbital_parity
            new_term = PauliLabel(
                frozenset(
                    tuple(i for i in term if i != (spin_orbital - 1, SinglePauli.Z))
                )
            )
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(new_term) is None:
                new_qubit_dict[new_term] = new_coefficient
            else:
                old_coefficient = new_qubit_dict.get(new_term)
                if old_coefficient is None:
                    old_coefficient = 0
                new_qubit_dict[new_term] = new_coefficient + old_coefficient
        else:
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(term) is None:
                new_qubit_dict[term] = coefficient
            else:
                old_coefficient = new_qubit_dict.get(term)
                if old_coefficient is None:
                    old_coefficient = 0
                new_qubit_dict[term] = coefficient + old_coefficient

    qubit_hamiltonian = Operator(new_qubit_dict)

    return qubit_hamiltonian


def remove_indices(symbolic_operator: Operator, indices: list[int]) -> Operator:
    """Returns the symbolic operator from which the operator with the specified
    index was removed.

    Args:
        symbolic_operator: An instance of the SymbolicOperator class.
        indices: A sequence of Int type object. The indices to be removed.

    Returns:
        The symbolic operator. The removed indices will be filled by shifting.
    """
    map: dict[int, int] = {}

    def new_index(index: int) -> int:
        if index in map:
            return map[index]
        map[index] = index - len(list(i for i in indices if (i - 1) < index))
        return map[index]

    def XYZ(index: int) -> str:
        if index == SinglePauli.X:
            return "X"
        if index == SinglePauli.Y:
            return "Y"
        if index == SinglePauli.Z:
            return "Z"
        return "I"

    new_operator = Operator()

    for term in symbolic_operator:
        if term == PAULI_IDENTITY:
            new_operator.add_term(PAULI_IDENTITY, symbolic_operator[term])
        else:
            new_term = ""
            for op in term:
                new_term += XYZ(op[1]) + str(new_index(op[0])) + " "
            new_operator.add_term(pauli_label(new_term), symbolic_operator[term])
    return new_operator
