# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, TypeVar, Union, cast

import numpy as np

from quri_parts.circuit.parameter_mapping import LinearParameterMapping
from quri_parts.circuit.parameter_shift import ShiftedParameters
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    Estimates,
    GradientEstimator,
)
from quri_parts.core.state import (
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
)

_ParametricStateT = TypeVar(
    "_ParametricStateT",
    bound=Union[ParametricCircuitQuantumState, ParametricQuantumStateVector],
)


@dataclass
class _Estimates:
    values: Sequence[complex]
    error_matrix: Optional[Sequence[Sequence[float]]]


def numerical_gradient_estimates(
    op: Estimatable,
    state: _ParametricStateT,
    params: Sequence[float],
    estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
    delta: float,
) -> Estimates[complex]:
    """Estimate a gradient of an expectation value of a given operator for a
    parametric state with respect to the state parameter by a numerical
    differentiation.

    The gradient estimates are configured with arguments as follows.

    Args:
        op: An operator of which expectation value is estimated.
        state: A parametric quantum state on which the operator
            expectation is evaluated.
        params: Parameter values for which the gradient is estimated.
        estimator: An estimator that estimates expectation values
            of the operator for the parametric states.
        delta: Step size for numerical differentiation.

    Returns:
        The estimated values (can be accessed with :attr:`.values`) with errors
        of estimation (can be accessed with :attr:`.error_matrix`).
    """

    v = []
    for i in range(len(params)):
        a = list(params)
        a[i] = params[i] + (delta * 0.5)
        v.append(a)
        a = list(params)
        a[i] = params[i] - (delta * 0.5)
        v.append(a)

    estimates = list(estimator(op, state, v))

    grad = []
    err_diag = []
    for i in range(len(params)):
        d = estimates[2 * i].value - estimates[2 * i + 1].value
        grad.append(d / delta)
        var = (estimates[2 * i].error ** 2) + (estimates[2 * i + 1].error ** 2)
        err_diag.append(np.sqrt(var) / delta)
    return _Estimates(grad, np.diag(err_diag).tolist())


def create_numerical_gradient_estimator(
    parametric_estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
    delta: float,
) -> GradientEstimator[_ParametricStateT]:
    """Create a :class:`GradientEstimator` that estimates gradient values.

    The gradient estimates are configured with arguments as follows.

    Args:
        parametric_estimator: An estimator that estimates expectation values
            of the operator for the parametric states.
        delta: Step size for numerical differentiation.
    """

    def estimator(
        operator: Estimatable, state: _ParametricStateT, params: Sequence[float]
    ) -> Estimates[complex]:
        return numerical_gradient_estimates(
            operator,
            state,
            params,
            parametric_estimator,
            delta,
        )

    return estimator


def parameter_shift_gradient_estimates(
    op: Estimatable,
    state: _ParametricStateT,
    params: Sequence[float],
    estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
) -> Estimates[complex]:
    """Estimate a gradient of an expectation value of a given operator for a
    parametric state with respect to the state parameter by the parameter shift
    rule.

    The gradient estimates are configured with arguments as follows.

    Args:
        op: An operator of which expectation value is estimated.
        state: A parametric quantum state on which the operator
            expectation is evaluated.
        params: Parameter values for which the gradient is estimated.
        estimator: An estimator that estimates expectation values
            of the operator for the parametric states.

    Returns:
        The estimated values (can be accessed with :attr:`.values`) with errors
        of estimation (can be accessed with :attr:`.error_matrix`).
    """
    param_mapping = cast(LinearParameterMapping, state.parametric_circuit.param_mapping)
    parameter_shift = ShiftedParameters(param_mapping)
    derivatives = parameter_shift.get_derivatives()
    shifted_params_and_coefs = [
        d.get_shifted_parameters_and_coef(params) for d in derivatives
    ]

    gate_params = set()
    for params_and_coefs in shifted_params_and_coefs:
        for p, _ in params_and_coefs:
            gate_params.add(p)
    gate_params_list = list(gate_params)

    raw_param_state = state.with_primitive_circuit()

    # When using a bound TypeVar, mypy raises an incompatible types error.
    # Therefore, after checking the instance type, cast to `_ParametricStateT`.
    if not (
        isinstance(raw_param_state, ParametricCircuitQuantumState)
        or isinstance(raw_param_state, ParametricQuantumStateVector)
    ):
        raise NotImplementedError(
            """
            Only the case that raw_param_state is _ParametricStateT
            has been implemented.
            """
        )

    estimates = estimator(
        op, cast(_ParametricStateT, raw_param_state), gate_params_list
    )
    estimates_dict = dict(zip(gate_params_list, estimates))

    grad = []
    err_diag = []
    for params_and_coefs in shifted_params_and_coefs:
        g = 0.0 + 0.0j
        var = 0.0
        for p, c in params_and_coefs:
            g += estimates_dict[p].value * c
            var += (estimates_dict[p].error ** 2) * abs(c) ** 2
        grad.append(g)
        err_diag.append(np.sqrt(var))

    return _Estimates(grad, np.diag(err_diag).tolist())


def create_parameter_shift_gradient_estimator(
    parametric_estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
) -> GradientEstimator[_ParametricStateT]:
    def estimator(
        operator: Estimatable, state: _ParametricStateT, params: Sequence[float]
    ) -> Estimates[complex]:
        return parameter_shift_gradient_estimates(
            operator, state, params, parametric_estimator
        )

    return estimator
