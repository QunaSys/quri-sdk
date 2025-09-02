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

from quri_parts.algo.mitigation.cdr import (  # noqa: E501
    cdr,
    create_cdr_estimator,
    create_exp_regression,
    create_exp_regression_with_const,
    create_exp_regression_with_const_log,
    create_polynomial_regression,
    make_training_circuits,
)
from quri_parts.circuit import (
    CZ,
    RX,
    RY,
    U2,
    U3,
    H,
    PauliRotation,
    QuantumCircuit,
    T,
    X,
    Z,
    is_clifford,
)
from quri_parts.core.operator.operator import Operator
from quri_parts.core.operator.pauli import pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

qubit_count = 4
cdr_gate_list = [
    H(0),
    X(2),
    Z(3),
    T(1),
    RX(1, 3.4 * np.pi),
    RX(1, 3.5 * np.pi),
    RY(2, 1.4 * np.pi),
    U2(1, 0.5 * np.pi, 1.7 * np.pi),
    U3(3, 0.5 * np.pi, 1.5 * np.pi, 2.5 * np.pi),
    CZ(2, 3),
    PauliRotation((0, 2, 3), (2, 3, 1), 0.51 * np.pi),
]
test_cdr_circuit = QuantumCircuit(qubit_count, gates=cdr_gate_list)


def test_make_training_circuis() -> None:
    with pytest.raises(ValueError):
        make_training_circuits(
            circuit=test_cdr_circuit,
            num_non_clifford_untouched=10,
            num_training_circuits=15,
            seed=10,
        )

    training_circuits = make_training_circuits(
        circuit=test_cdr_circuit,
        num_non_clifford_untouched=3,
        num_training_circuits=15,
        seed=10,
    )
    clifford_gate_list = [
        H(0),
        X(2),
        Z(3),
        RX(1, 3.5 * np.pi),
        U2(1, 0.5 * np.pi, 1.5 * np.pi),
        CZ(2, 3),
        PauliRotation((0, 2, 3), (2, 3, 1), 0.5 * np.pi),
    ]
    clifford_cdr_circuit = QuantumCircuit(qubit_count, gates=clifford_gate_list)
    with pytest.raises(ValueError):
        make_training_circuits(
            circuit=clifford_cdr_circuit,
            num_non_clifford_untouched=3,
            num_training_circuits=15,
            seed=10,
        )

    assert len(training_circuits) == 15

    for training_circuit in training_circuits:
        assert (
            len([gate for gate in training_circuit.gates if not is_clifford(gate)]) == 3
        )


def test_create_polynomial_regression() -> None:
    ans_value = -10.542988
    noise_value = 0.117
    x_data = [1, 2.6, 4.4, 7.8, 9.1]
    y_data: list[float] = [3, 13, 35, 44, 56]
    regression_method = create_polynomial_regression(order=3)
    assert np.isclose(regression_method(noise_value, x_data, y_data), ans_value)


def test_create_exp_regression() -> None:
    ans_value = 4.70268507
    noise_value = 0.117
    x_data = [0.1, 0.6, 0.8, 1.1, 1.3, 1.4, 1.7]
    y_data: list[float] = [
        4.62015,
        9.68224,
        14.87231,
        32.77924,
        61.14104,
        85.91254,
        266.55352,
    ]
    exp_regression = create_exp_regression(order=2)
    assert np.isclose(exp_regression(noise_value, x_data, y_data), ans_value)

    const = 0.6
    exp_with_const_nolog = create_exp_regression_with_const(order=2, constant=const)
    assert np.isclose(exp_with_const_nolog(noise_value, x_data, y_data), ans_value)

    exp_with_const_log = create_exp_regression_with_const_log(order=2, constant=const)
    assert np.isclose(exp_with_const_log(noise_value, x_data, y_data), ans_value)


test_operator = Operator({pauli_label("X0 Y1 Z2"): 1.0, pauli_label("Y1 Z2 X3"): 1.3})


def test_cdr() -> None:
    test_noisy_estimator = create_qulacs_vector_concurrent_estimator()
    test_exact_estimator = create_qulacs_vector_concurrent_estimator()
    regression_method = create_polynomial_regression(order=2)

    with pytest.raises(NotImplementedError):
        cdr(
            obs=Operator({pauli_label("X0 Y1 Z2"): 1.5j}),
            circuit=test_cdr_circuit,
            noisy_estimator=test_noisy_estimator,
            exact_estimator=test_exact_estimator,
            regression_method=regression_method,
            num_training_circuits=10,
            fraction_of_replacement=0.1,
            seed=10,
        )

        with pytest.raises(ValueError):
            cdr(
                obs=Operator({pauli_label("X0 Y1 Z2"): 1.5j}),
                circuit=test_cdr_circuit,
                noisy_estimator=test_noisy_estimator,
                exact_estimator=test_exact_estimator,
                regression_method=regression_method,
                num_training_circuits=10,
                fraction_of_replacement=2,
                seed=10,
            )

    mitigated_value = cdr(
        obs=test_operator,
        circuit=test_cdr_circuit,
        noisy_estimator=test_noisy_estimator,
        exact_estimator=test_exact_estimator,
        regression_method=regression_method,
        num_training_circuits=10,
        fraction_of_replacement=0.1,
        seed=10,
    )
    assert np.isclose(mitigated_value, 0.0092313)


def test_create_cdr_estimator() -> None:
    test_noisy_estimator = create_qulacs_vector_concurrent_estimator()
    test_exact_estimator = create_qulacs_vector_concurrent_estimator()
    regression_method = create_polynomial_regression(order=2)
    cdr_estimator = create_cdr_estimator(
        noisy_estimator=test_noisy_estimator,
        exact_estimator=test_exact_estimator,
        regression_method=regression_method,
        num_training_circuits=10,
        fraction_of_replacement=0.1,
        seed=10,
    )
    test_state = GeneralCircuitQuantumState(qubit_count, test_cdr_circuit)
    cdr_value_estimated = cdr_estimator(test_operator, test_state)

    mitigated_value = cdr(
        obs=test_operator,
        circuit=test_cdr_circuit,
        noisy_estimator=test_noisy_estimator,
        exact_estimator=test_exact_estimator,
        regression_method=regression_method,
        num_training_circuits=10,
        fraction_of_replacement=0.1,
        seed=10,
    )
    assert np.isclose(cdr_value_estimated.value, mitigated_value)
