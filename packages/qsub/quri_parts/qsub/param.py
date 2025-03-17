from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", np.int64, np.float64, np.complex128)


@dataclass(frozen=True)
class ArrayRef(Generic[T]):
    """An object that holds a reference to an array.

    It is provided to avoid unnecessary copying when large numerical arrays are used
    as the Param of Op. Equivalence between ArrayRef instances is determined by the
    instance ids of the internal arrays.

    Examples:
        .. highlight:: python
        .. code-block:: python

            import numpy as np

            xs = np.arange(5)
            ys = xs
            zs = np.arange(5)

            a = ArrayRef(xs)
            b = ArrayRef(ys)
            c = ArrayRef(zs)

            assert a == b
            assert a != c

            d = a
            e = copy.copy(d)
            f = copy.deepcopy(e)

            assert a == d
            assert a == e
            assert a != f
    """

    array: npt.NDArray[T]

    def __hash__(self) -> int:
        return id(self.array)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayRef):
            return False
        return id(self.array) == id(other.array)

    def __str__(self) -> str:
        return str(self.array)
