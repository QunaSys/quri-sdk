# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Iterable, cast

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FittedResult:
    """An immutable (frozen) dataclass representing a fitted result."""

    #: Coefficients of the fitting function, obtained by fitting.
    parameters: list[float]

    #: The value at a certain point obtained by using a fitted function.
    value: float


def polynomial_fitting(
    x_data: Iterable[float], y_data: Iterable[float], order: int, point: float
) -> FittedResult:
    """Polynomial fitting with an polynomial of a given order.

    Args:
        x_data : x-coordinates for fitting.
        y_data : y-coordinates for fitting.
        order: Order of the polynomial used for fitting.
        point: A point at which the fitted function to be evaluated.

    Returns:
        The fitted parameters (can be accessed with :attr:`.parameters`) and a value
        at the input point of the fitted function (can be accessed with :attr:`.value`).
    """

    fitted_params = list(Polynomial.fit(x_data, y_data, order).convert().coef)
    fitted_value = cast(float, Polynomial(fitted_params)(point))
    return FittedResult(parameters=fitted_params, value=fitted_value)


def exp_fitting(
    x_data: Iterable[float],
    y_data: Iterable[float],
    order: int,
    point: float,
) -> FittedResult:
    """Curve fitting with an exponential ansatz f(x) = a + b exp(p(x)), where p(x) is a
    polynomial of a given order.

    Args:
        x_data : x-coordinates for fitting.
        y_data : y-coordinates for fitting.
        order: Order of the polynomial on the exponential used for fitting.
        point: A point at which the fitted function to be evaluated.

    Returns:
        The fitted parameters (can be accessed with :attr:`.parameters`) and a value
        at the input point of the fitted function (can be accessed with :attr:`.value`).
    """

    if order > len(list(x_data)) - 1:
        raise ValueError(
            "The order cannot be larger than the number of len(x_data) -1. "
        )

    init_param = [0.0] * (order + 3)

    def model_function(x: float, *coeffs: float) -> float:
        return cast(
            float,
            coeffs[0] + coeffs[1] * np.exp(Polynomial(coeffs[2:])(x)),
        )

    fitted_params, _ = curve_fit(model_function, x_data, y_data, init_param)
    fitted_value = fitted_params[0] + fitted_params[1] * np.exp(
        Polynomial(fitted_params[2:])(point)
    )
    return FittedResult(parameters=fitted_params, value=fitted_value)


def exp_fitting_with_const(
    x_data: Iterable[float],
    y_data: Iterable[float],
    order: int,
    constant: float,
    point: float,
) -> FittedResult:
    """Curve fitting with an exponential ansatz f(x) = constant + b exp(p(x)), where
    p(x) is a polynomial of a given order and constant is a known parameter (obtained
    as the infinite limit f(x->inf) when f(x) converges to a finite asymptotic value).

    Args:
        x_data : x-coordinates for fitting.
        y_data : y-coordinates for fitting.
        order: Order of the polynomial on the exponential used for fitting.
        constant: A constant deduced from asymptotic behavior f(x->inf).
        point: A point at which the fitted function to be evaluated.

    Returns:
        The fitted parameters (can be accessed with :attr:`.parameters`) and a value
        at the input point of the fitted function (can be accessed with :attr:`.value`).
    """

    if order > len(list(x_data)) - 1:
        raise ValueError(
            "The order cannot be larger than the number of len(x_data) -1. "
        )

    init_param = [0.0] * (order + 2)

    def model_function_with_constant(x: float, *coeffs: float) -> float:
        return cast(float, constant + coeffs[0] * np.exp(Polynomial(coeffs[1:])(x)))

    fitted_params, _ = curve_fit(
        model_function_with_constant, x_data, y_data, init_param
    )
    fitted_value = constant + fitted_params[0] * np.exp(
        Polynomial(fitted_params[1:])(point)
    )
    return FittedResult(parameters=fitted_params, value=fitted_value)


def exp_fitting_with_const_log(
    x_data: Iterable[float],
    y_data: Iterable[float],
    order: int,
    constant: float,
    point: float,
) -> FittedResult:
    """Log fitting with an exponential ansatz f(x) = constant + b exp(p(x)), where p(x)
    is a polynomial of a given order and constant is a known parameter (obtained as the
    infinite limit f(x->inf) when f(x) converges to a finite asymptotic value).

    Args:
        x_data : x-coordinates for fitting.
        y_data : y-coordinates for fitting.
        order: Order of the polynomial on the exponential used for fitting.
        constant: A constant deduced from asymptotic behavior f(x->inf).
        point: A point at which the fitted function to be evaluated.

    Returns:
        The fitted parameters (can be accessed with :attr:`.parameters`) and a value
        at the input point of the fitted function (can be accessed with :attr:`.value`).
    """

    if order > len(list(x_data)) - 1:
        raise ValueError(
            "The order cannot be larger than the number of len(x_data) -1. "
        )

    linear_param = Polynomial.fit(x_data, y_data, 1).convert().coef[1]
    sign = np.sign(linear_param)

    regular_eps = 1.0e-8
    shifted_value = [max(abs(value - constant), regular_eps) for value in y_data]
    log_values = np.log(shifted_value)

    fitted_params = (
        Polynomial.fit(
            x_data,
            log_values,
            deg=order,
        )
        .convert()
        .coef
    )
    fitted_value = constant + sign * np.exp(Polynomial(fitted_params[0:])(point))
    return FittedResult(parameters=fitted_params, value=fitted_value)
