# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Sequence
from unittest.mock import Mock

import numpy as np
import pytest

from quri_parts.circuit import (
    LinearMappedParametricQuantumCircuit,
    ParametricQuantumCircuit,
)
from quri_parts.core.estimator import Estimatable, Estimate
from quri_parts.core.estimator.gradient import (
    _ParametricStateT,
    create_numerical_gradient_estimator,
    create_parameter_shift_gradient_estimator,
    numerical_gradient_estimates,
    parameter_shift_gradient_estimates,
)
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label, zero
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
)


def a_state() -> ParametricCircuitQuantumState:
    circuit = ParametricQuantumCircuit(1)
    circuit.add_Z_gate(0)
    circuit.add_ParametricRX_gate(0)
    s = ParametricCircuitQuantumState(1, circuit)
    return s


def b_state() -> ParametricCircuitQuantumState:
    circuit = ParametricQuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_X_gate(1)
    circuit.add_ParametricRY_gate(0)
    circuit.add_ParametricRY_gate(1)
    s = ParametricCircuitQuantumState(2, circuit)
    return s


def c_state() -> ParametricCircuitQuantumState:
    circuit = LinearMappedParametricQuantumCircuit(1)
    theta1 = circuit.add_parameter("theta1")
    circuit.add_ParametricRX_gate(0, {theta1: 1 / 2})
    return ParametricCircuitQuantumState(1, circuit)


def d_state() -> ParametricCircuitQuantumState:
    circuit = LinearMappedParametricQuantumCircuit(2)
    theta1, theta2 = circuit.add_parameters("theta1", "theta2")
    circuit.add_H_gate(0)
    circuit.add_X_gate(1)
    circuit.add_ParametricRY_gate(0, {theta1: 1})
    circuit.add_ParametricRY_gate(1, {theta2: 1})
    return ParametricCircuitQuantumState(2, circuit)


class _Estimate:
    @property
    def value(self) -> complex:
        return 0.0

    @property
    def error(self) -> float:
        return 2.0


def estimator(
    op: Estimatable, state: _ParametricStateT, v: Sequence[float]
) -> Iterable[Estimate[complex]]:
    return [_Estimate(), _Estimate()]


def mock_estimator() -> Mock:
    mock = Mock()
    mock.side_effect = estimator
    return mock


class TestGradientEstimator:
    def test_create_numerical_gradient_estimator(self) -> None:
        init_params = [0.0]
        delta = 1e-6
        estimator = mock_estimator()
        numerical_estimator = create_numerical_gradient_estimator(estimator, delta)
        estimates = numerical_estimator(zero(), a_state(), init_params)
        estimator.assert_called_once()
        assert estimates.values == [0.0]
        error_expected = np.sqrt((2.0**2 + 2.0**2) / delta**2)
        assert estimates.error_matrix is not None
        assert estimates.error_matrix[0][0] == pytest.approx(error_expected)

    def test_create_parameter_shift_gradient_estimator(self) -> None:
        init_params = [0.0]
        estimator = mock_estimator()
        parameter_shift_estimator = create_parameter_shift_gradient_estimator(estimator)
        estimates = parameter_shift_estimator(zero(), c_state(), init_params)
        estimator.assert_called_once()
        assert estimates.values == [0.0]
        error_expected = np.sqrt(2.0**2 * 0.25**2 + 2.0**2 * 0.25**2)
        assert estimates.error_matrix is not None
        assert estimates.error_matrix[0][0] == pytest.approx(error_expected)


class TestNumericalGradientEstimate:
    def test_zero_op(self) -> None:
        init_params = [0.0]
        delta = 1e-6
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = numerical_gradient_estimates(
            zero(), a_state(), init_params, estimator, delta
        )
        assert estimates.values == [0.0]
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0]])

    def test_const_op(self) -> None:
        op = Operator({PAULI_IDENTITY: 3.0})
        init_params = [0.0]
        delta = 1e-6
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = numerical_gradient_estimates(
            op, a_state(), init_params, estimator, delta
        )
        assert estimates.values == [0.0]
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0]])

    def test_gradient_estimator(self) -> None:
        op = Operator({pauli_label("Z0"): 2.0})
        init_params = [np.pi * 0.25]
        delta = 1e-6
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = numerical_gradient_estimates(
            op, a_state(), init_params, estimator, delta
        )
        assert estimates.values[0] == pytest.approx(-1.41421356)
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0]])

    def test_multi_params_gradient_estimator(self) -> None:
        op = Operator({pauli_label("Z0 X1"): 2.0})
        init_params = [np.pi * 0.5, np.pi * 0.25]
        delta = 1e-6
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = numerical_gradient_estimates(
            op, b_state(), init_params, estimator, delta
        )
        assert estimates.values == [0.0, pytest.approx(1.41421356)]
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0, 0.0], [0.0, 0.0]])

    def test_delta(self) -> None:
        op = Operator({pauli_label("Z0"): 2.0})
        init_params = [np.pi * 0.25]
        real_value = -1.41421356237
        delta1 = 1e-2
        delta2 = 1e-3
        delta3 = 1e-4
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates1 = numerical_gradient_estimates(
            op, a_state(), init_params, estimator, delta1
        )
        estimates2 = numerical_gradient_estimates(
            op, a_state(), init_params, estimator, delta2
        )
        estimates3 = numerical_gradient_estimates(
            op, a_state(), init_params, estimator, delta3
        )
        assert abs(estimates1.values[0] - real_value) < 1e-4
        assert abs(estimates2.values[0] - real_value) < 1e-4
        assert abs(estimates3.values[0] - real_value) < 1e-4


class TestParameterShiftGradientEstimate:
    def test_zero_op(self) -> None:
        init_params = [0.0]
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = parameter_shift_gradient_estimates(
            zero(), c_state(), init_params, estimator
        )
        assert estimates.values == [0.0]
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0]])

    def test_const_op(self) -> None:
        op = Operator({PAULI_IDENTITY: 3.0})
        init_params = [0.0]
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = parameter_shift_gradient_estimates(
            op, c_state(), init_params, estimator
        )
        assert estimates.values == [0.0]
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0]])

    def test_pauli_Z_op(self) -> None:
        op = Operator({pauli_label("Z0"): 2.0})
        init_params = [np.pi * 0.5]
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = parameter_shift_gradient_estimates(
            op, c_state(), init_params, estimator
        )
        assert estimates.values[0] == pytest.approx(-0.70710678)
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0]])

    def test_multi_params_gradient_estimator(self) -> None:
        op = Operator({pauli_label("Z0 X1"): 2.0})
        init_params = [np.pi * 0.5, np.pi * 0.25]
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = parameter_shift_gradient_estimates(
            op, d_state(), init_params, estimator
        )
        assert estimates.values[0] == pytest.approx(0.0)
        assert estimates.values[1] == pytest.approx(1.41421356)
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0, 0.0], [0.0, 0.0]])

    def test_primitive_unbound_parametric_circuit(self) -> None:
        op = Operator({pauli_label("Z0 X1"): 2.0})
        init_params = [np.pi * 0.5, np.pi * 0.25]
        estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimates = parameter_shift_gradient_estimates(
            op, b_state(), init_params, estimator
        )
        assert estimates.values == [0.0, pytest.approx(1.41421356)]
        assert estimates.error_matrix is not None
        np.testing.assert_allclose(estimates.error_matrix, [[0.0, 0.0], [0.0, 0.0]])
