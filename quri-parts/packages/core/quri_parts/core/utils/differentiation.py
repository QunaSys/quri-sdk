# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Callable, Protocol, Sequence, TypeVar, Union

from typing_extensions import TypeAlias

from quri_parts.core.operator import Operator, truncate

#: Represents a function that generates :class:`Operator` from given
#: parameters, e.g. generates molecular Hamiltonian from coordinates
#: of atoms.
OperatorGenerator: TypeAlias = Callable[[Sequence[float]], Operator]

#: Represents a function that calculates the gradients of an operator
#: at given parameters.
OperatorGradientCalculator: TypeAlias = Callable[
    [Sequence[float], OperatorGenerator], Sequence[Operator]
]

#: Represents a function that calculates the hessian of an operator
#: at given parameters.
OperatorHessianCalculator: TypeAlias = Callable[
    [Sequence[float], OperatorGenerator], Sequence[Sequence[Operator]]
]

_T = TypeVar("_T")


class DifferentiableObjectProtocol(Protocol):
    r""".. document private functions.

    .. automethod:: __add__

    .. automethod:: __sub__

    .. automethod:: __truediv__
    """

    @abstractmethod
    def __add__(self: _T, other: _T) -> _T:
        ...

    @abstractmethod
    def __sub__(self: _T, other: _T) -> _T:
        ...

    @abstractmethod
    def __truediv__(self: _T, other: Union[int, float, complex]) -> _T:
        ...


T = TypeVar("T", bound=DifferentiableObjectProtocol)


def forward_difference_gradient_formula(
    f: Callable[[Sequence[float]], T],
    params: Sequence[float],
    step: float = 1e-5,
) -> Sequence[T]:
    """Returns the gradient of a passed function from `params` with two-point
    forward-difference formula."""
    dim = len(params)
    forward_points = [
        [params[k] + step * (k == i) for k in range(dim)] for i in range(dim)
    ]
    f_vals_forward = [f(forward) for forward in forward_points]
    f_val_orig = f(params)

    return [(f_val_fw - f_val_orig) / step for f_val_fw in f_vals_forward]


def backward_difference_gradient_formula(
    f: Callable[[Sequence[float]], T],
    params: Sequence[float],
    step: float = 1e-5,
) -> Sequence[T]:
    """Returns the gradient of a passed function from `params` with two-point
    backward-difference formula."""
    dim = len(params)
    backward_points = [
        [params[k] - step * (k == i) for k in range(dim)] for i in range(dim)
    ]
    f_vals_backward = [f(backward) for backward in backward_points]
    f_val_orig = f(params)

    return [(f_val_orig - f_val_bw) / step for f_val_bw in f_vals_backward]


def central_difference_gradient_formula(
    f: Callable[[Sequence[float]], T],
    params: Sequence[float],
    step: float = 1e-5,
) -> Sequence[T]:
    """Returns the gradient of a passed function from `params` with central-
    difference formula."""
    dim = len(params)
    forward_points = [
        [params[k] + step * (k == i) for k in range(dim)] for i in range(dim)
    ]
    backward_points = [
        [params[k] - step * (k == i) for k in range(dim)] for i in range(dim)
    ]
    f_vals_forward = [f(forward) for forward in forward_points]
    f_vals_backward = [f(backward) for backward in backward_points]

    return [
        (f_val_fw - f_val_bw) / (2 * step)
        for f_val_fw, f_val_bw in zip(f_vals_forward, f_vals_backward)
    ]


def forward_difference_hessian_formula(
    f: Callable[[Sequence[float]], T],
    params: Sequence[float],
    step: float = 1e-5,
) -> Sequence[Sequence[T]]:
    """Returns the hessian of a passed function from `params` with forward-
    difference formula."""
    dim = len(params)
    points_pp = [
        [
            [params[k] + step * (k == i) + step * (k == j) for k in range(dim)]
            for j in range(dim)
        ]
        for i in range(dim)
    ]
    points_p = [[params[k] + step * (k == i) for k in range(dim)] for i in range(dim)]
    f_vals_pp = [[f(point_ij) for point_ij in points_i] for points_i in points_pp]
    f_vals_p = [f(point_i) for point_i in points_p]
    f_val_orig = f(params)

    ret: list[list[T]] = [[] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            ret[i].append(
                (f_vals_pp[i][j] - f_vals_p[i] - f_vals_p[j] + f_val_orig)
                / (step * step)
            )

    return ret


def backward_difference_hessian_formula(
    f: Callable[[Sequence[float]], T],
    params: Sequence[float],
    step: float = 1e-5,
) -> Sequence[Sequence[T]]:
    """Returns the hessian of a passed function from `params` with backward-
    difference formula."""
    dim = len(params)
    points_mm = [
        [
            [params[k] - step * (k == i) - step * (k == j) for k in range(dim)]
            for j in range(dim)
        ]
        for i in range(dim)
    ]
    points_m = [[params[k] - step * (k == i) for k in range(dim)] for i in range(dim)]
    f_vals_mm = [[f(point_ij) for point_ij in points_i] for points_i in points_mm]
    f_vals_m = [f(point_i) for point_i in points_m]
    f_val_orig = f(params)

    ret: list[list[T]] = [[] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            ret[i].append(
                (f_val_orig - f_vals_m[i] - f_vals_m[j] + f_vals_mm[i][j])
                / (step * step)
            )

    return ret


def central_difference_hessian_formula(
    f: Callable[[Sequence[float]], T],
    params: Sequence[float],
    step: float = 1e-5,
) -> Sequence[Sequence[T]]:
    """Returns the hessian of a passed function from `params` with central-
    difference formula."""
    dim = len(params)
    points_pp = [
        [
            [params[k] + step * (k == i) + step * (k == j) for k in range(dim)]
            for j in range(dim)
        ]
        for i in range(dim)
    ]
    points_pm = [
        [
            [params[k] + step * (k == i) - step * (k == j) for k in range(dim)]
            for j in range(dim)
        ]
        for i in range(dim)
    ]
    points_mp = [
        [
            [params[k] - step * (k == i) + step * (k == j) for k in range(dim)]
            for j in range(dim)
        ]
        for i in range(dim)
    ]
    points_mm = [
        [
            [params[k] - step * (k == i) - step * (k == j) for k in range(dim)]
            for j in range(dim)
        ]
        for i in range(dim)
    ]

    f_vals_pp = [[f(point_ij) for point_ij in points_i] for points_i in points_pp]
    f_vals_pm = [[f(point_ij) for point_ij in points_i] for points_i in points_pm]
    f_vals_mp = [[f(point_ij) for point_ij in points_i] for points_i in points_mp]
    f_vals_mm = [[f(point_ij) for point_ij in points_i] for points_i in points_mm]

    ret: list[list[T]] = [[] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            ret[i].append(
                (f_vals_pp[i][j] - f_vals_pm[i][j] - f_vals_mp[i][j] + f_vals_mm[i][j])
                / (4 * step * step)
            )

    return ret


gradient = central_difference_gradient_formula
hessian = central_difference_hessian_formula


def numerical_operator_gradient(
    params: Sequence[float],
    operator_generator: OperatorGenerator,
    difference_formula: Callable[
        [Callable[[Sequence[float]], Operator], Sequence[float], float],
        Sequence[Operator],
    ] = gradient,
    step: float = 1e-5,
    atol: float = 1e-8,
) -> Sequence[Operator]:
    """Function that returns the numerical gradient of an :class:`Operator`
    with respect to the operator parameters.

    Args:
        params: Parameters at which the gradient is calculated.
        operator_generator: :class:`OperatorGenerator`.
        difference_formula: Method to calculate gradients.
        step: Step size for ``difference_formula``.
        atol: Absolute tolerance. Terms whose coefficients are smaller than ``atol``
            will be ignored.
    """

    ops = [
        truncate(op, atol)
        for op in difference_formula(operator_generator, params, step)
    ]
    return ops


def numerical_operator_hessian(
    params: Sequence[float],
    operator_generator: OperatorGenerator,
    difference_formula: Callable[
        [Callable[[Sequence[float]], Operator], Sequence[float], float],
        Sequence[Sequence[Operator]],
    ] = hessian,
    step: float = 1e-5,
    atol: float = 1e-8,
) -> Sequence[Sequence[Operator]]:
    """Function that returns the numerical hessian of an :class:`Operator` with
    respect to the operator parameters.

    Args:
        params: Parameters at which the hessian is calculated.
        operator_generator: :class:`OperatorGenerator`.
        difference_formula: Method to calculate hessian.
        step: Step size for ``difference_formula``.
        atol: Absolute tolerance. Terms whose coefficients are smaller than ``atol``
            will be ignored.
    """

    ops = [
        [truncate(op, atol) for op in ops]
        for ops in difference_formula(operator_generator, params, step)
    ]

    return ops
