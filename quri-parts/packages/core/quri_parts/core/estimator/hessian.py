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
from typing import Optional, Sequence, TypeVar, Union, cast

import numpy as np

from quri_parts.circuit import LinearParameterMapping
from quri_parts.circuit.parameter_shift import ShiftedParameters
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    Estimatable,
    HessianEstimator,
    MatrixEstimates,
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
class _MatrixEstimates:
    values: Sequence[Sequence[complex]]
    error_tensor: Optional[Sequence[Sequence[Sequence[Sequence[float]]]]]


def parameter_shift_hessian_estimates(
    op: Estimatable,
    state: _ParametricStateT,
    params: Sequence[float],
    estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
) -> MatrixEstimates[complex]:
    """Estimate a hessian of an expectation value of a given operator for a
    parametric state with respect to the state parameter by the parameter shift
    rule.

    The hessian estimates are configured with arguments as follows.

    Args:
        op: An operator of which expectation value is estimated.
        state: A parametric quantum state on which the operator
            expectation is evaluated.
        params: Parameter values for which the hessian is estimated.
        estimator: An estimator that estimates expectation values
            of the operator for the parametric states.

    Returns:
        The estimated values (can be accessed with :attr:`.values`) with errors
        of estimation (can be accessed with :attr:`.error_tensor`). Currently,
        :attr:`.error_tensor` returns `None`.
    """
    param_circuit = state.parametric_circuit
    param_mapping = cast(LinearParameterMapping, param_circuit.param_mapping)
    parameter_shift = ShiftedParameters(param_mapping)
    derivatives = [
        derivs_i.get_derivatives() for derivs_i in parameter_shift.get_derivatives()
    ]
    shifted_params_and_coeffs_list = [
        [deriv.get_shifted_parameters_and_coef(params) for deriv in derivs]
        for derivs in derivatives
    ]
    raw_param_state = cast(_ParametricStateT, state.with_primitive_circuit())
    uniq_g_params = set()
    for shifted_params_and_coeffs in shifted_params_and_coeffs_list:
        for params_and_coefs in shifted_params_and_coeffs:
            for p, _ in params_and_coefs:
                uniq_g_params.add(p)
    uniq_g_params_list = list(uniq_g_params)

    # Estimate the expectation values
    estimates = estimator(op, raw_param_state, uniq_g_params_list)
    estimates_dict = dict(zip(uniq_g_params_list, estimates))

    # Sum up the expectation values with the coefficients multiplied
    hessian = np.zeros((len(derivatives), len(derivatives)), dtype=np.complex128)
    for i in range(len(derivatives)):
        for j in range(len(derivatives)):
            g = 0.0 + 0.0j
            for p, c in shifted_params_and_coeffs_list[i][j]:
                g += estimates_dict[p].value * c
            hessian[i, j] = g

    return _MatrixEstimates(hessian.tolist(), None)


def create_parameter_shift_hessian_estimator(
    parametric_estimator: ConcurrentParametricQuantumEstimator[_ParametricStateT],
) -> HessianEstimator[_ParametricStateT]:
    def estimator(
        operator: Estimatable, state: _ParametricStateT, params: Sequence[float]
    ) -> MatrixEstimates[complex]:
        return parameter_shift_hessian_estimates(
            operator, state, params, parametric_estimator
        )

    return estimator
