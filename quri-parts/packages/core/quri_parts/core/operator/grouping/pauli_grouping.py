# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt

from ..operator import Operator
from ..pauli import PAULI_IDENTITY, CommutablePauliSet, PauliLabel, SinglePauli
from ..representation import (
    BinarySymplecticVector,
    bsv_bitwise_commute,
    pauli_label_to_bsv,
)


def individual_pauli_grouping(
    paulis: Union[Operator, Iterable[PauliLabel]]
) -> Iterable[CommutablePauliSet]:
    """An implementation of :class:`~PauliGrouping`, which creates a group for
    each of the PauliLabel's (i.e. no grouping)."""
    if isinstance(paulis, Operator):
        paulis = paulis.keys()
    return frozenset({frozenset({p}) for p in paulis})


@dataclass
class _Group:
    bsv: BinarySymplecticVector
    paulis: list[PauliLabel] = field(default_factory=list)


def bitwise_pauli_grouping(
    paulis: Union[Operator, Iterable[PauliLabel]]
) -> Iterable[CommutablePauliSet]:
    """An implementation of :class:`~PauliGrouping`, which groups the
    PauliLabel's based on bitwise commutability.

    In this implementation, first the PauliLabel's with only X, Y or Z
    are assigned to special groups (all X group, all Y group and all Z
    group). After that each PauliLabel is assigned to a group with
    greedy algorithm, meaning that it assigned to the first compatible
    group, i.e. all PauliLabel's in the group bitwise-commute with the
    given PauliLabel.
    """
    if isinstance(paulis, Operator):
        paulis = paulis.keys()

    identity: list[PauliLabel] = []
    all_X: list[PauliLabel] = []
    all_Y: list[PauliLabel] = []
    all_Z: list[PauliLabel] = []
    groups: list[_Group] = []
    for pauli in paulis:
        if pauli == PAULI_IDENTITY:
            identity.append(pauli)
            continue

        has_X = False
        has_Y = False
        has_Z = False
        for _, p in pauli:
            if p == SinglePauli.X:
                has_X = True
            if p == SinglePauli.Y:
                has_Y = True
            if p == SinglePauli.Z:
                has_Z = True
            if (has_X and has_Y) or (has_Y and has_Z) or (has_Z and has_X):
                break
        if has_X and (not has_Y) and (not has_Z):
            all_X.append(pauli)
            continue
        if (not has_X) and has_Y and (not has_Z):
            all_Y.append(pauli)
            continue
        if (not has_X) and (not has_Y) and has_Z:
            all_Z.append(pauli)
            continue

        bsv = pauli_label_to_bsv(pauli)
        _add_pauli_to_groups(pauli, bsv, groups)

    pauli_groups = [frozenset(group.paulis) for group in groups]
    if identity:
        pauli_groups.append(frozenset(identity))
    if all_X:
        pauli_groups.append(frozenset(all_X))
    if all_Y:
        pauli_groups.append(frozenset(all_Y))
    if all_Z:
        pauli_groups.append(frozenset(all_Z))

    return frozenset(pauli_groups)


def sorted_injection_grouping(
    paulis: Union[Operator, Iterable[PauliLabel]]
) -> Iterable[CommutablePauliSet]:
    """An implementation of :class:`~PauliGrouping`, which groups the
    PauliLabel's based on the sorted injection algorithm.

    For more details refer https://arxiv.org/abs/1908.06942.

    The core of the algorithm is to start grouping after sorting the operator
    list by the descending order of absolute value of their coefficients.
    The grouping is of operators that bitwise-commute with all the other
    operators in the group.
    """
    if isinstance(paulis, Operator):
        coefficients: npt.NDArray[np.complex128] = np.array(list(paulis.values()))
        paulis = np.array(list(paulis.keys()))

        # Sorting paulis and coefficients based on abs values of coefficients
        # in descending order.
        inds = np.argsort(np.abs(coefficients))[::-1]
        paulis = paulis[inds]

    else:
        paulis = list(paulis)

    groups: list[_Group] = []

    for pauli in paulis:
        bsv = pauli_label_to_bsv(pauli)
        _add_pauli_to_groups(pauli, bsv, groups)

    pauli_groups = [frozenset(group.paulis) for group in groups]

    return frozenset(pauli_groups)


def _add_pauli_to_groups(
    pauli: PauliLabel, bsv: BinarySymplecticVector, groups: list[_Group]
) -> None:
    for group in groups:
        if bsv_bitwise_commute(bsv, group.bsv):
            group.paulis.append(pauli)
            # This new binary symplectic vector has all the non-identity Pauli matrices
            # contained in the PauliLabel's in the group.
            # Therefore you can check commutability with all the PauliLabel's in the
            # group by checking commutablity with this binary symplec vector.
            group.bsv = BinarySymplecticVector(
                x=(bsv.x | group.bsv.x),
                z=(bsv.z | group.bsv.z),
                phase=bsv.phase * group.bsv.phase,
            )
            return

    group = _Group(bsv=bsv)
    group.paulis.append(pauli)
    groups.append(group)
