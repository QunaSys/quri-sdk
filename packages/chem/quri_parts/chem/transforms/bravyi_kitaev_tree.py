# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

from quri_parts.chem.transforms.fenwick_tree import FenwickTree
from quri_parts.chem.transforms.fermion import FermionLabel
from quri_parts.chem.transforms.fermion_operator import FermionOperator
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label


def bravyi_kitaev_tree(operator: FermionOperator, n_qubits: int) -> Operator:
    """Transform FermionOperator to Operator using Bravyi-Kitaev mapping.

    Implementation from arxiv:1701.07072
    """
    if isinstance(operator, FermionOperator):
        fenwick_tree = FenwickTree(n_qubits)
        transformed_terms = (
            _transform_operator_term(
                term=label, coefficient=coef, fenwick_tree=fenwick_tree
            )
            for label, coef in operator.items()
        )
        return inline_sum(summands=transformed_terms, seed=Operator())
    raise TypeError("Operator must be a FermionOperator, ")


def _transform_operator_term(
    term: FermionLabel, coefficient: complex, fenwick_tree: FenwickTree
) -> Operator:
    transformed_ladder_ops = (
        _transform_ladder_operator(ladder_operator, fenwick_tree)
        for ladder_operator in term
    )
    return inline_product(
        factors=transformed_ladder_ops, seed=Operator({PAULI_IDENTITY: coefficient})
    )


def _transform_ladder_operator(
    ladder_operator: tuple[int, int], fenwick_tree: FenwickTree
) -> Operator:
    index = ladder_operator[0]

    # Parity set. Set of nodes to apply Z to.
    parity_set = [
        node.index
        for node in fenwick_tree.get_parity_set(index)
        if node.index is not None
    ]

    # Update set. Set of ancestors to apply X to.
    update_set = [
        node.index
        for node in fenwick_tree.get_update_set(index)
        if node.index is not None
    ]

    # The C(j) set.
    remainder_set = [
        node.index
        for node in fenwick_tree.get_remainder_set(index)
        if node.index is not None
    ]

    # Switch between lowering/raising operators.
    d_coefficient = -0.5j if ladder_operator[1] else 0.5j

    # The fermion lowering operator is given by
    # a = (c+id)/2 where c, d are the majoranas.
    d_majorana_str = f"Y{ladder_operator[0]}"
    for index in remainder_set:
        d_majorana_str += f" Z{index}"
    d_majorana_component = Operator(
        {
            pauli_label(d_majorana_str): d_coefficient,
        }
    )
    # update_setが空の場合は、Xの係数が0になるので、ここで終了
    if update_set:
        d_majorana_str = ""
        for index in update_set:
            d_majorana_str += f" X{index}"
        d_majorana_component *= Operator(
            {
                pauli_label(d_majorana_str): 1.0,
            }
        )

    c_majorana_str = f"X{ladder_operator[0]}"
    for index in parity_set:
        c_majorana_str += f" Z{index}"
    c_majorana_component = Operator(
        {
            pauli_label(c_majorana_str): 0.5,
        }
    )
    if update_set:
        c_majorana_str = ""
        for index in update_set:
            c_majorana_str += f" X{index}"
        c_majorana_component *= Operator(
            {
                pauli_label(c_majorana_str): 1.0,
            }
        )
    return c_majorana_component + d_majorana_component


def inline_product(factors: Iterable[Operator], seed: Operator) -> Operator:
    """Computes a product, using the __imul__ operator.

    Args:
        seed (T): The starting total. The unit value.
        factors (iterable[T]): Values to multiply (with *=) into the total.
    Returns:
        T: The result of multiplying all the factors into the unit value.
    """
    for r in factors:
        seed *= r
    return seed


def inline_sum(summands: Iterable[Operator], seed: Operator) -> Operator:
    """Computes a sum, using the __iadd__ operator.

    Args:
        seed (T): The starting total. The zero value.
        summands (iterable[T]): Values to add (with +=) into the total.
    Returns:
        T: The result of adding all the factors into the zero value.
    """
    for r in summands:
        seed += r
    return seed
