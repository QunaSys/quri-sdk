from typing import Sequence, TypeVar, Union, cast

import numpy as np
from numpy.typing import ArrayLike

Numerics = TypeVar("Numerics", int, float, complex)


def cast_to_list(int_sequence: Union[Sequence[Numerics], ArrayLike]) -> list[Numerics]:
    """Cast a sequence of numerics (int, float, complex) or an array to a list
    of the same type.

    This is a workaround for too strict type annotation of Qulacs
    """
    return cast(list[Numerics], np.asarray(int_sequence).reshape(-1).tolist())
