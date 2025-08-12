# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from abc import abstractmethod
from collections.abc import Collection, Iterable, Mapping, Sequence, Set
from enum import IntEnum
from typing import Optional, Protocol, Union, cast, runtime_checkable
from weakref import WeakValueDictionary

from typing_extensions import TypeAlias


class SinglePauli(IntEnum):
    """An integer enumeration representing Pauli matrices X, Y, Z acting on a
    single qubit."""

    X = 1
    Y = 2
    Z = 3


_pauli_name_map = {p.value: p.name for p in SinglePauli}

_pauli_products_map: dict[tuple[int, int], Optional[tuple[int, complex]]] = {
    (SinglePauli.X, SinglePauli.X): None,
    (SinglePauli.X, SinglePauli.Y): (SinglePauli.Z, 1.0j),
    (SinglePauli.X, SinglePauli.Z): (SinglePauli.Y, -1.0j),
    (SinglePauli.Y, SinglePauli.X): (SinglePauli.Z, -1.0j),
    (SinglePauli.Y, SinglePauli.Y): None,
    (SinglePauli.Y, SinglePauli.Z): (SinglePauli.X, 1.0j),
    (SinglePauli.Z, SinglePauli.X): (SinglePauli.Y, 1.0j),
    (SinglePauli.Z, SinglePauli.Y): (SinglePauli.X, -1.0j),
    (SinglePauli.Z, SinglePauli.Z): None,
}

_pauli_cache: WeakValueDictionary[str, "PauliLabel"] = WeakValueDictionary()


def pauli_name(p: int) -> str:
    """Returns the name of Pauli matrix for a SinglePauli (int)"""
    # Using a predefined lookup dict for performance (faster than SinglePauli(p).name)
    return _pauli_name_map[p]


def _parse_pauli_label_str(s: str) -> Mapping[int, SinglePauli]:
    # Remove spaces after XYZ: "X 0 Y 1 Z 2" -> "X0 Y1 Z2"
    terms = re.sub(r"([XYZ])\s*", r"\1", s).split()
    if len(terms) == 0:
        raise ValueError(f"No valid Pauli label found in '{s}'")
    d: dict[int, SinglePauli] = dict()
    for t in terms:
        m = re.fullmatch(r"([XYZ])([0-9]+)", t)
        if not m:
            raise ValueError(f"Invalid Pauli label: '{t}'")
        index = int(m.group(2))
        if d.get(index) is not None:
            raise ValueError(f"Duplicate qubit index: {index}")
        d[index] = SinglePauli[m.group(1)]
    return d


@runtime_checkable
class PauliLabelProvider(Protocol):
    """PauliLabelProvider is an interface for classes providing information of
    a Pauli label."""

    @abstractmethod
    def get_index_list(self) -> Collection[int]:
        """A list of indices of qubits on which the Pauli matrices act."""
        ...

    @abstractmethod
    def get_pauli_id_list(self) -> Collection[int]:
        """A list of SinglePauli's contained in the Pauli string, sorted in the
        order corresponding to the qubit index list."""
        ...


class PauliLabel(frozenset[tuple[int, int]]):
    """PauliLabel represents a label for a Pauli string, e.g. ``X0 Y1 Z2``.

    It represents only a "label" of a multi-qubit Pauli term and does not hold a
    coefficient or a phase factor.
    This class is basically a :class:`~frozenset` of pairs of a qubit index and
    a SinglePauli, and also can be used as a key for a dict (`Hashable`_).

    Example:
        >>> label = pauli_label({(0, SinglePauli.X), (1, SinglePauli.Y), \
(2, SinglePauli.Z)})
        >>> label
        PauliLabel({(0, <SinglePauli.X: 1>), (1, <SinglePauli.Y: 2>), \
(2, <SinglePauli.Z: 3>)})
        >>> str(label)
        'X0 Y1 Z2'
        >>> for pair in label:
        ...     print(pair)
        ...
        (0, <SinglePauli.X: 1>)
        (1, <SinglePauli.Y: 2>)
        (2, <SinglePauli.Z: 3>)
        >>> label.pauli_at(1)
        <SinglePauli.Y: 2>
        >>> 1 in label.qubit_indices()
        True
        >>> for i in label.qubit_indices():
        ...     print(f"index: {i}, operator: {label.pauli_at(i)}")
        ...
        index: 0, operator: 1
        index: 1, operator: 2
        index: 2, operator: 3

    You can construct a PauliLabel in one of the following ways:

    * With a tuples of a qubit index (int) and a Pauli matrix label (int)
        >>> pauli_label({(0, 1), (1, 2), (2, 3)})
    * With a tuples of a qubit index (int) and a Pauli matrix label (SinglePauli)
        >>> pauli_label({(0, SinglePauli.X), (1, SinglePauli.Y), (2, SinglePauli.Z)})
    * With a string representation of a Pauli string (Note that string parsing has \
      a significant performance overhead)
        >>> pauli_label("X0 Y1 Z2")
    * From a qubit index list and a matrix label (SinglePauli) list
        >>> PauliLabel.from_index_and_pauli_list(
                [0, 1, 2], [SinglePauli.X, SinglePauli.Y, SinglePauli.Z]
            )
    * From an object providing a qubit index list and a matrix label list \
      (:class:`~PauliLabelProvider`)
        >>> pauli_label(obj) # obj has get_index_list and get_pauli_id_list methods

    Note that the following code does not raise an error but creates
    an invalid PauliLabel. To avoid this kind of error, it is recommended to use
    a factory function :func:`~pauli_label` instead of the original constructor
    :func:`~PauliLabel`.

    >>> PauliLabel("X0 Y1 Z2") # Should be pauli_label("X0 Y1 Z2") or \
PauliLabel.from_str("X0 Y1 Z2")
    PauliLabel({' ', 'Z', '0', '1', 'X', 'Y', '2'})

    The string input should be given as a list of single pauli terms separated by white
    spaces. Each single pauli term consists of a Pauli matrix label (``X``, ``Y`` or
    ``Z``) and a qubit index (int), possibly separated by white spaces.

    Valid string input examples:

    * ``X0 Y1 Z2``
    * ``X 0 Y 1 Z 2``

    Invalid string input examples:

    * ``X0 Y1 A2`` (Invalid Pauli matrix label ``A``)
    * ``X0Y1Z2`` (No spaces between each term)
    * ``X0 Y1 Z1`` (Duplicate qubit indices)
    * ``X0 Y Z2`` (No qubit index specified)
    * ``X0 1 Z2`` (No Pauli matrix label specified)

    Order of the terms is irrelevant when comparing two PauliLabel's:

    >>> pauli_label("X0 Y1 Z2") == pauli_label("X0 Z2 Y1")
    True

    Performance tip: An index access by :meth:`~PauliLabel.pauli_at` involves
    an iteration inside it, so it is inefficient when you want to iterate over
    all single Pauli terms in a PauliLabel. Instead of doing this:

    >>> for i in range(len(label)):
    ...     p = label.pauli_at(i)
    ...     # Do something with p

    the following is faster:

    >>> for i, p in label:
    ...     # Do something with p

    .. _Hashable:
        https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable
    """

    def __new__(cls, arg: Iterable[tuple[int, int]] = ()) -> "PauliLabel":
        instance = super().__new__(cls, arg)  # type: ignore
        pl_str = str(instance)
        if pl_str in _pauli_cache:
            return _pauli_cache[pl_str]
        else:
            _pauli_cache[pl_str] = instance
            return instance

    def __str__(self) -> str:
        if len(self) == 0:
            return "I"
        return " ".join(
            [SinglePauli(o).name + str(i) for i, o in sorted(self, key=lambda t: t[0])]
        )

    def qubit_indices(self) -> Collection[int]:
        """Returns a Collection of qubit indices on which Pauli matrices act.

        Note that this collection does not have a definite ordering.
        """
        return list(map(lambda t: t[0], self))

    def pauli_at(self, index: int) -> Optional[int]:
        """Returns a Pauli matrix at the qubit index ``index``.

        Returns ``None`` if no Pauli matrix acts on the index.
        """
        for i, o in self:
            if i == index:
                return o
        return None

    @property
    def index_and_pauli_id_list(self) -> tuple[Sequence[int], Sequence[int]]:
        """A pair of a list of indices of qubits on which the Pauli matrices
        act, and a list of the Pauli matrices in the order corresponding to the
        index list."""
        index_list, pauli_list = zip(*self)
        return list(index_list), list(pauli_list)

    @staticmethod
    def from_index_and_pauli_list(
        index: Collection[int], pauli: Collection[int]
    ) -> "PauliLabel":
        """Create a PauliLabel from a list of qubit indices and a list of Pauli
        matrices."""
        if len(index) != len(pauli):
            raise ValueError("Length of index and pauli unmatch")
        return PauliLabel(zip(index, pauli))

    @staticmethod
    def from_str(pauli_label_str: str) -> "PauliLabel":
        """Create a PauliLabel from a string representation of a Pauli
        String."""
        map = _parse_pauli_label_str(pauli_label_str)
        return PauliLabel(map.items())

    @staticmethod
    def of(pauli: PauliLabelProvider) -> "PauliLabel":
        """Create a PauliLabel from an object providing a qubit index list and
        a SinglePauli list."""
        return PauliLabel.from_index_and_pauli_list(
            pauli.get_index_list(), pauli.get_pauli_id_list()
        )


def pauli_label(
    p: Union[str, PauliLabelProvider, Iterable[tuple[int, int]]]
) -> PauliLabel:
    """Create a PauliLabel.

        The argument can be one of the followings:

        * A string representation of a Pauli string.
        * A :class:`~PauliLabelProvider`, an object providing a qubit index list and \
    a SinglePauli list.
        * An Iterable of tuple[int, int], pairs of a qubit index and a SinglePauli.
    """
    if isinstance(p, str):
        return PauliLabel.from_str(p)
    elif isinstance(p, PauliLabelProvider):
        return PauliLabel.of(p)
    elif isinstance(p, Iterable):
        p = cast(Iterable[tuple[int, int]], p)
        return PauliLabel(p)
    else:
        raise ValueError(f"Invalid argument for pauli_label: {p}")


def pauli_product(pauli1: PauliLabel, pauli2: PauliLabel) -> tuple[PauliLabel, complex]:
    """Returns the product of two PauliLabels.

    Note that this function returns a tuple (PauliLabel, phase) since the
    multiplication of two Pauli operators can yield a Pauli operator multiplied by
    the imaginary unit (e.g., XY = iZ).
    """
    p_label_dict = dict(pauli1)
    phase: complex = 1.0
    for index, pauli in pauli2:
        if index in p_label_dict:
            prod = _pauli_products_map[(p_label_dict[index], pauli)]
            if prod is None:
                del p_label_dict[index]
                continue
            p_label_dict[index] = prod[0]
            phase *= prod[1]
        else:
            p_label_dict[index] = pauli
    return PauliLabel(p_label_dict.items()), phase


PAULI_IDENTITY = PauliLabel()
"""PauliLabel used as an identity."""


#: CommutablePauliSet
CommutablePauliSet: TypeAlias = Set[PauliLabel]
