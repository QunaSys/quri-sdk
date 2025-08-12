# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence, Union

import numpy as np
import pytest
from typing_extensions import TypeAlias

from quri_parts.chem.properties import create_energy_gradient_estimator
from quri_parts.circuit import ParametricQuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import (
    ComputationalBasisState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
)
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

ParamState: TypeAlias = Union[
    ParametricCircuitQuantumState, ParametricQuantumStateVector
]


def _h_generator(params: Sequence[float]) -> Operator:
    return Operator(
        {
            PAULI_IDENTITY: 1.0 * params[0],
            pauli_label("Z0"): 2.0 * params[0] * params[1],
            pauli_label("Z1"): 3.0 * params[2] ** 2,
            pauli_label("Z2"): 4.0 * params[3] ** 3,
            pauli_label("Z0 Z1"): 5.0 * (params[4] - 1.0),
            pauli_label("Z0 Z2"): 6.0 * (params[5] - params[4]),
        }
    )


def test_energy_gradient_estimator() -> None:
    qubit_count = 3
    h_params = [1, 2, 3, 4, 5, 6]
    estimator = create_qulacs_vector_concurrent_estimator()
    energy_grad_estimator: Callable[
        [ParamState, Sequence[float]], Sequence[float]
    ] = create_energy_gradient_estimator(estimator, h_params, _h_generator)

    # no circuit parameters
    param_circuit = ParametricQuantumCircuit(qubit_count)
    param_circuit.extend(ComputationalBasisState(qubit_count, bits=0b111).circuit)
    param_state = ParametricCircuitQuantumState(qubit_count, param_circuit)
    expected = [-3.0, -2.0, -18.0, -192.0, -1.0, 6.0]
    assert np.allclose(energy_grad_estimator(param_state, []), expected)

    # add parametric gates
    param_circuit.add_ParametricRX_gate(0)
    param_circuit.add_H_gate(1)
    param_circuit.add_ParametricRY_gate(2)
    param_state = ParametricCircuitQuantumState(qubit_count, param_circuit)
    circuit_params = [np.pi, np.pi]
    expected = [5.0, 2.0, 0.0, 192.0, -6.0, 6.0]
    assert np.allclose(energy_grad_estimator(param_state, circuit_params), expected)


def test_energy_gradient_estimator_non_hermitian_op() -> None:
    def _non_hermitian_op_generator(params: Sequence[float]) -> Operator:
        return Operator(
            {
                PAULI_IDENTITY: 1.0 * params[0],
                pauli_label("Z0"): 2.0j * params[0] * params[1],
                pauli_label("Z1"): 3.0 * params[2] ** 2,
                pauli_label("Z2"): 4.0 * params[3] ** 3,
                pauli_label("Z0 Z1"): 5.0 * (params[4] - 1.0),
                pauli_label("Z0 Z2"): 6.0 * (params[5] - params[4]),
            }
        )

    h_params = [1, 2, 3, 4, 5, 6]
    estimator = create_qulacs_vector_concurrent_estimator()
    with pytest.raises(ValueError):
        create_energy_gradient_estimator(
            estimator, h_params, _non_hermitian_op_generator
        )
