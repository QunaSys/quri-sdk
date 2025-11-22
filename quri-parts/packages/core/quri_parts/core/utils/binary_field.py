# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from collections.abc import Iterable, Iterator, Sequence, Sized
from itertools import chain
from typing import Union, overload


class BinaryArray(Sequence[int]):
    """Array with elements in the binary field.

    ``+``, ``*`` and ``@`` operators are defined. ``+`` and ``*``
    perform element wise addition and multiplication respectively,
    returning a :class:`~BinaryArray`. ``@`` performs binary inner
    product between two :class:`~BinaryArray`, returning either 0 or 1.
    """

    def __init__(self, iter: Iterable[Union[int, bool]]):
        b = 0
        length = 0
        for index, val in enumerate(iter):
            if not (isinstance(val, bool) or isinstance(val, int)):
                raise TypeError(
                    f"Invalid value: {val}. either boolean or integer is expected."
                )
            if val:
                b |= 1 << index
            length = max(index + 1, length)
        self._b = b
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({list(self[i] for i in range(self._length))})"
        )

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[int, Sequence[int]]:
        if isinstance(index, int):
            return self._b >> index & 1
        else:
            indices = list(self)[index]
            return BinaryArray(indices)

    def __setitem__(self, index: int, value: int) -> None:
        current_value = self[index]
        if current_value != value:
            self._b ^= 1 << index

    def __iter__(self) -> Iterator[int]:
        for i in range(self._length):
            yield self[i]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryArray):
            return False
        return self._b == other._b and self._length == other._length

    def __add__(self, other: "BinaryArray") -> "BinaryArray":
        copied = copy.deepcopy(self)
        copied += other
        return copied

    def __iadd__(self, other: "BinaryArray") -> "BinaryArray":
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Invalid type: {other.__class__.__name__}, "
                f"{self.__class__.__name__} is expected."
            )
        if self._length != other._length:
            raise ValueError("The length of vector must be same.")
        self._b ^= other._b
        return self

    def __mul__(self, other: "BinaryArray") -> "BinaryArray":
        copied = copy.deepcopy(self)
        copied *= other
        return copied

    def __imul__(self, other: "BinaryArray") -> "BinaryArray":
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Invalid type: {other.__class__.__name__}, "
                f"{self.__class__.__name__} is expected."
            )
        if self._length != other._length:
            raise ValueError("The length of vector must be same.")
        self._b &= other._b
        return self

    def __matmul__(self, other: "BinaryArray") -> int:
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Invalid type: {other.__class__.__name__}, "
                f"{self.__class__.__name__} is expected."
            )
        return sum(self * other) % 2

    @property
    def binary(self) -> int:
        """Binary representation of the array."""
        return self._b


class BinaryMatrix(Iterable[BinaryArray], Sized):
    """Matrix with elements in the binary field.

    ``@`` operator is defined as matrix multiplication (in the binary
    field). It can be applied to either of a :class:`~BinaryMatrix` or
    :class:`~BinaryArray`.
    """

    def __init__(self, iter: Iterable[Iterable[Union[bool, int]]]):
        rows: list[BinaryArray] = []
        for column_index, row in enumerate(iter):
            row_vector = BinaryArray(row)
            if column_index > 0:
                if len(rows[0]) != len(row_vector):
                    raise ValueError("The length of all the rows must be the same.")
            rows.append(row_vector)

        self._rows = rows
        self._l = len(rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryMatrix):
            return False
        if self._l == other._l:
            return all(self._rows[i] == other._rows[i] for i in range(self._l))
        return False

    def __len__(self) -> int:
        return self._l

    def __str__(self) -> str:
        return self.__repr__()

    @overload
    def __getitem__(self, index: int) -> BinaryArray:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> int:
        ...

    def __getitem__(
        self, index: Union[int, tuple[int, int]]
    ) -> Union[BinaryArray, int]:
        if isinstance(index, int):
            return self._rows[index]
        i, j = index
        return self._rows[i][j]

    @overload
    def __setitem__(self, index: int, value: BinaryArray) -> None:
        ...

    @overload
    def __setitem__(self, index: tuple[int, int], value: int) -> None:
        ...

    def __setitem__(
        self, index: Union[int, tuple[int, int]], value: Union[BinaryArray, int]
    ) -> None:
        if isinstance(index, int):
            if not isinstance(value, BinaryArray):
                raise ValueError("value should be a BinaryArray")
            self._rows[index] = value
        else:
            if not isinstance(value, int):
                raise ValueError("value should be an int")
            i, j = index
            self._rows[i][j] = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({list(list(b for b in r) for r in self._rows)})"
        )

    def __iter__(self) -> Iterator[BinaryArray]:
        for r in self._rows:
            yield r

    @overload
    def __matmul__(self, other: BinaryArray) -> BinaryArray:
        ...

    @overload
    def __matmul__(self, other: "BinaryMatrix") -> "BinaryMatrix":
        ...

    def __matmul__(self, other: object) -> Union[BinaryArray, "BinaryMatrix"]:
        if isinstance(other, BinaryArray):
            return BinaryArray(r @ other for r in self._rows)
        elif isinstance(other, self.__class__):
            ret = []
            other_transposed = other.transpose()
            for row in self:
                ret.append(
                    list(row @ other_column for other_column in other_transposed)
                )
            return self.__class__(ret)
        raise TypeError(
            f"Invalid type: {other.__class__.__name__}, "
            f"either BinaryArray or BinaryMatrix is expected."
        )

    def transpose(self) -> "BinaryMatrix":
        """Returns a transposed matrix."""
        return self.__class__(list(zip(*self._rows)))


def hstack(a: BinaryMatrix, b: BinaryMatrix) -> BinaryMatrix:
    if len(a) != len(b):
        raise ValueError("The size of matrices must be same.")
    return BinaryMatrix(chain(a._rows[i], b._rows[i]) for i in range(len(a)))


def vstack(a: BinaryMatrix, b: BinaryMatrix) -> BinaryMatrix:
    return hstack(a.transpose(), b.transpose()).transpose()


def inverse(mat: BinaryMatrix) -> BinaryMatrix:
    """Returns an inverse of the matrix."""
    n = len(mat)
    eye = BinaryMatrix((i == j for j in range(n)) for i in range(n))
    mat_aug = hstack(mat, eye)

    for j in range(n):
        for i in range(j, n):
            if mat_aug[i, j] == 1:
                if i != j:
                    mat_aug[i], mat_aug[j] = mat_aug[j], mat_aug[i]
                pivot_row = mat_aug[j]
                break
        for i in range(j, n):
            val = 1 if i == j else 0
            if mat_aug[i, j] != val:
                mat_aug[i] += pivot_row
    for j in range(n - 1, -1, -1):
        for i in range(n - 1, j - 1, -1):
            if mat_aug[i, j] == 1:
                pivot_row = mat_aug[i]
                break
        for i in range(j - 1, -1, -1):
            val = 1 if i == j else 0
            if mat_aug[i, j] != val:
                mat_aug[i] += pivot_row

    return BinaryMatrix(list(row)[n:] for row in mat_aug)
