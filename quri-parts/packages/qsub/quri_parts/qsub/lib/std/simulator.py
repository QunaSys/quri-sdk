import numpy as np
import numpy.typing as npt

from quri_parts.qsub.op import ParamUnitaryDef, param_op

from . import NS


class _Matrix(ParamUnitaryDef[float]):
    ns = NS
    name = "Matrix"

    def qubit_count_fn(self, qubit_count: int, matrix: np.array) -> int:
        return qubit_count


Matrix = param_op(_Matrix)


class _Permutation(ParamUnitaryDef[int, npt.NDArray]):
    ns = NS
    name = "Permutation"

    def qubit_count_fn(self, qubit_count: int, permutation: npt.NDArray) -> int:
        return qubit_count


Permutation = param_op(_Permutation)


class _GeneralizedPermutation(ParamUnitaryDef[int, npt.NDArray, npt.NDArray]):
    ns = NS
    name = "GeneralizedPermutation"

    def qubit_count_fn(
        self, qubit_count: int, permutation: npt.NDArray, diagonals: npt.NDArray
    ) -> int:
        return qubit_count


GeneralizedPermutation = param_op(_GeneralizedPermutation)
