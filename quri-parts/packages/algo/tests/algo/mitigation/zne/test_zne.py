# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from quri_parts.algo.mitigation.zne.zne import (  # noqa: E501
    _get_residual_n_gates,
    create_exp_extrapolate,
    create_exp_extrapolate_with_const,
    create_exp_extrapolate_with_const_log,
    create_folding_left,
    create_folding_random,
    create_folding_right,
    create_polynomial_extrapolate,
    create_zne_estimator,
    richardson_extrapolation,
    scaling_circuit_folding,
    zne,
)
from quri_parts.circuit import CNOT, CZ, RX, RY, SWAP, H, QuantumCircuit, Z
from quri_parts.core.operator.operator import Operator
from quri_parts.core.operator.pauli import pauli_label
from quri_parts.core.state.state import GeneralCircuitQuantumState
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

qubit_count = 4
scale_factor = 2.6
gate_list = [H(0), Z(1), RX(0, 3.4 * np.pi), RY(2, 1.4 * np.pi), CNOT(2, 3), SWAP(1, 2)]
test_circuit = QuantumCircuit(qubit_count, gates=gate_list)


def test_get_residual_n_gates() -> None:
    assert _get_residual_n_gates(test_circuit, scale_factor) == 4


def test_create_folding_left() -> None:
    folding_method = create_folding_left()
    assert folding_method(test_circuit, scale_factor) == [0, 1, 2, 3]


def test_create_folding_right() -> None:
    folding_method = create_folding_right()
    assert folding_method(test_circuit, scale_factor) == [2, 3, 4, 5]


def test_create_folding_random() -> None:
    folding_method = create_folding_random(seed=0)
    assert folding_method(test_circuit, scale_factor) == [2, 3, 1, 0]


def test_scaling_circuit() -> None:
    folding_method = create_folding_left()
    scaling_circuit = scaling_circuit_folding(
        test_circuit, scale_factor, folding_method
    )

    gate_list = [
        H(0),
        H(0),
        H(0),
        Z(1),
        Z(1),
        Z(1),
        RX(0, 3.4 * np.pi),
        RX(0, -3.4 * np.pi),
        RX(0, 3.4 * np.pi),
        RY(2, 1.4 * np.pi),
        RY(2, -1.4 * np.pi),
        RY(2, 1.4 * np.pi),
        CNOT(2, 3),
        SWAP(1, 2),
    ]
    circuit = QuantumCircuit(qubit_count, gates=gate_list)

    assert scaling_circuit.gates == circuit.gates


def test_create_polynomial_extrapolate() -> None:
    ans_param = 0.3
    scale_factors = [1, 2.6, 4.4, 7.8, 9.1]
    exp_values: list[float] = [30.9, 22.9512, -172.0392, -1534.6776, -2550.2298]
    extrapolate_method = create_polynomial_extrapolate(order=3)
    assert np.isclose(extrapolate_method(scale_factors, exp_values), ans_param)


def test_create_exp_extrapolate() -> None:
    ans_param = 0.6 + 1.2 * np.exp(1.1)
    scale_factors = [0.1, 0.6, 0.8, 1.1, 1.3, 1.4, 1.7]
    exp_values: list[float] = [
        3.961951,
        4.596909,
        6.850875,
        22.544610,
        82.028192,
        185.371801,
        4532.613299,
    ]

    exp_without_const = create_exp_extrapolate(order=3)
    assert np.isclose(exp_without_const(scale_factors, exp_values), ans_param)

    const = 0.6
    exp_with_const_nolog = create_exp_extrapolate_with_const(order=3, constant=const)
    assert np.isclose(exp_with_const_nolog(scale_factors, exp_values), ans_param)

    exp_with_const_log = create_exp_extrapolate_with_const_log(order=3, constant=const)
    assert np.isclose(exp_with_const_log(scale_factors, exp_values), ans_param)


test_operator = Operator({pauli_label("X0 Y1 Z2"): 1.5, pauli_label("Z1 X2 Y3"): 2.3})
zne_gate_list = [
    H(0),
    RX(1, 3.4 * np.pi),
    RX(0, 3.4 * np.pi),
    RY(2, 1.4 * np.pi),
    CZ(2, 3),
]
test_zne_circuit = QuantumCircuit(qubit_count, gates=zne_gate_list)
test_scale_factors = [1, 6.2, 8, 11, 13.2, 14, 17]


def test_zne() -> None:
    test_estimator = create_qulacs_vector_concurrent_estimator()
    extrapolate_method = create_polynomial_extrapolate(order=3)
    folding_method = create_folding_left()

    with pytest.raises(NotImplementedError):
        zne(
            obs=Operator({pauli_label("X0 Y1 Z2"): 1.5j}),
            circuit=test_zne_circuit,
            estimator=test_estimator,
            scale_factors=test_scale_factors,
            extrapolate_method=extrapolate_method,
            folding_method=folding_method,
        )

    extrapolated = zne(
        obs=test_operator,
        circuit=test_zne_circuit,
        estimator=test_estimator,
        scale_factors=test_scale_factors,
        extrapolate_method=extrapolate_method,
        folding_method=folding_method,
    )
    assert np.isclose(-0.4408393, extrapolated)


def test_richardson_extrapolation() -> None:
    test_estimator = create_qulacs_vector_concurrent_estimator()
    folding_method = create_folding_left()

    extrapolated_richardson = richardson_extrapolation(
        obs=test_operator,
        circuit=test_zne_circuit,
        estimator=test_estimator,
        scale_factors=test_scale_factors,
        folding_method=folding_method,
    )

    extrapolate_method = create_polynomial_extrapolate(
        order=len(test_scale_factors) - 1
    )
    extrapolated_zne = zne(
        obs=test_operator,
        circuit=test_zne_circuit,
        estimator=test_estimator,
        scale_factors=test_scale_factors,
        extrapolate_method=extrapolate_method,
        folding_method=folding_method,
    )

    assert np.isclose(extrapolated_richardson, extrapolated_zne)


def test_create_zne_estimator() -> None:
    test_estimator = create_qulacs_vector_concurrent_estimator()
    extrapolate_method = create_polynomial_extrapolate(order=3)
    folding_method = create_folding_left()
    zne_estimator = create_zne_estimator(
        estimator=test_estimator,
        scale_factors=test_scale_factors,
        extrapolate_method=extrapolate_method,
        folding_method=folding_method,
    )

    test_state = GeneralCircuitQuantumState(qubit_count, test_zne_circuit)
    zne_value = zne_estimator(test_operator, test_state)

    extrapolated = zne(
        obs=test_operator,
        circuit=test_zne_circuit,
        estimator=test_estimator,
        scale_factors=test_scale_factors,
        extrapolate_method=extrapolate_method,
        folding_method=folding_method,
    )
    assert np.isclose(zne_value.value, extrapolated)
