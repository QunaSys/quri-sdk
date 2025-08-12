# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from typing import Sequence

import numpy as np

from quri_parts.circuit import (
    CONST,
    LinearMappedParametricQuantumCircuit,
    ParametricQuantumCircuit,
)
from quri_parts.core.estimator import Estimatable, MatrixEstimates, _ParametricStateT
from quri_parts.core.estimator.hessian import create_parameter_shift_hessian_estimator
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
)


class TestParameterShiftHessianEstimate(unittest.TestCase):
    state: ParametricCircuitQuantumState

    @classmethod
    def setUpClass(cls) -> None:
        param_circuit = ParametricQuantumCircuit(3)
        param_circuit.add_X_gate(0)
        param_circuit.add_X_gate(1)
        param_circuit.add_X_gate(2)
        param_circuit.add_ParametricRX_gate(0)
        param_circuit.add_ParametricRY_gate(1)

        cls.state = ParametricCircuitQuantumState(3, param_circuit)

    def hessian_estimator(
        self, operator: Estimatable, state: _ParametricStateT, params: Sequence[float]
    ) -> MatrixEstimates[complex]:
        parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimator = create_parameter_shift_hessian_estimator(parametric_estimator)
        return estimator(operator, state, params)

    def test_hessian(self) -> None:
        operator = Operator(
            {
                PAULI_IDENTITY: 1.0,
                pauli_label("Z0"): 4.0,
                pauli_label("Z1"): 3.0 * 3**2,
                pauli_label("Z2"): 4.0 * 4**3,
                pauli_label("Z0 Z1"): 5.0 * (5 - 1),
                pauli_label("Z0 Z2"): 6.0,
            }
        )

        params = [np.pi / 3, np.pi / 7]

        hessian_matrix = self.hessian_estimator(operator, self.state, params)
        expected = [[-10.0097, 7.51509], [7.51509, 15.3165]]
        assert np.allclose(hessian_matrix.values, expected)

    def test_zero_op_hessian(self) -> None:
        operator = Operator()

        params = [np.pi / 3, np.pi / 7]
        hessian_matrix = self.hessian_estimator(operator, self.state, params)
        expected = [[0, 0], [0, 0]]
        assert np.allclose(hessian_matrix.values, expected)

    def test_identity_op_hessian(self) -> None:
        operator = Operator({PAULI_IDENTITY: 8})

        params = [np.pi / 3, np.pi / 7]
        hessian_matrix = self.hessian_estimator(operator, self.state, params)
        expected = [[0, 0], [0, 0]]
        assert np.allclose(hessian_matrix.values, expected)


class TestParameterShiftHessianEstimateLinearMappedState(unittest.TestCase):
    state: ParametricCircuitQuantumState

    @classmethod
    def setUpClass(cls) -> None:
        param_circuit = LinearMappedParametricQuantumCircuit(3)
        param_circuit.add_X_gate(0)
        param_circuit.add_X_gate(1)
        param_circuit.add_X_gate(2)
        theta1, theta2 = param_circuit.add_parameters("theta1", "theta2")
        param_circuit.add_ParametricRX_gate(
            0, angle={theta1: 1, theta2: 1 / 3, CONST: -np.pi / 7}
        )
        param_circuit.add_ParametricRX_gate(
            2, angle={theta1: 1 / 8, theta2: -1 / 3, CONST: np.pi / 11}
        )
        param_circuit.add_ParametricRY_gate(
            1, angle={theta1: -1 / 4, theta2: 1, CONST: np.pi / 3}
        )

        cls.state = ParametricCircuitQuantumState(3, param_circuit)

    def hessian_estimator(
        self, operator: Estimatable, state: _ParametricStateT, params: Sequence[float]
    ) -> MatrixEstimates[complex]:
        parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
        estimator = create_parameter_shift_hessian_estimator(parametric_estimator)
        return estimator(operator, state, params)

    def test_hessian(self) -> None:
        operator = Operator(
            {
                PAULI_IDENTITY: 1.0,
                pauli_label("Z0"): 4.0,
                pauli_label("Z1"): 3.0 * 3**2,
                pauli_label("Z2"): 4.0 * 4**3,
                pauli_label("Z0 Z1"): 5.0 * (5 - 1),
                pauli_label("Z0 Z2"): 6.0,
            }
        )

        params = [np.pi / 3, np.pi / 7]
        hessian_matrix = self.hessian_estimator(operator, self.state, params)
        expected = [[-8.25677, -1.72578], [-1.72578, 38.6785]]
        assert np.allclose(hessian_matrix.values, expected)

    def test_zero_op_hessian(self) -> None:
        operator = Operator()

        params = [np.pi / 3, np.pi / 7]
        hessian_matrix = self.hessian_estimator(operator, self.state, params)
        expected = [[0, 0], [0, 0]]
        assert np.allclose(hessian_matrix.values, expected)

    def test_identity_op_hessian(self) -> None:
        operator = Operator({PAULI_IDENTITY: 8})

        params = [np.pi / 3, np.pi / 7]
        hessian_matrix = self.hessian_estimator(operator, self.state, params)
        expected = [[0, 0], [0, 0]]
        assert np.allclose(hessian_matrix.values, expected)
