# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from numpy import allclose, array, isclose, pi

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    QuantumStateVector,
)
from quri_parts.qulacs.circuit import compile_circuit
from quri_parts.qulacs.simulator import (
    evaluate_state_to_vector,
    get_marginal_probability,
    run_circuit,
)


def test_evaluate_state_to_vector_general_circuit_state_to_vec() -> None:
    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = GeneralCircuitQuantumState(n, quantum_circuit)
    quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_evaluate_state_to_vector_vec_to_vec() -> None:
    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = QuantumStateVector(n, array([1, 0, 0, 0]), quantum_circuit)
    quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_evaluate_state_to_vector_comp_state_to_vec() -> None:
    circuit_quantum_state = ComputationalBasisState(4, bits=0b0101)
    quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array([0] * 5 + [1] + [0] * 10)

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_run_circuit_simple() -> None:
    n = 2
    theta_x = pi / 7
    theta_y = pi / 8

    update_circuit = QuantumCircuit(n)
    update_circuit.add_CNOT_gate(0, 1)
    update_circuit.add_X_gate(0)
    update_circuit.add_RX_gate(0, theta_x)
    update_circuit.add_RY_gate(0, theta_y)
    update_circuit.add_H_gate(1)

    updated_quantum_state_vector = run_circuit(update_circuit, array([1, 0, 0, 0]))

    # expectation
    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert allclose(updated_quantum_state_vector, expect_output_vector)


def test_run_circuit_simple_with_compiled_circuit() -> None:
    n = 2
    theta_x = pi / 7
    theta_y = pi / 8

    update_circuit = QuantumCircuit(n)
    update_circuit.add_CNOT_gate(0, 1)
    update_circuit.add_X_gate(0)
    update_circuit.add_RX_gate(0, theta_x)
    update_circuit.add_RY_gate(0, theta_y)
    update_circuit.add_H_gate(1)

    compiled_circuit = compile_circuit(update_circuit)

    updated_quantum_state_vector = run_circuit(compiled_circuit, array([1, 0, 0, 0]))

    # expectation
    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert allclose(updated_quantum_state_vector, expect_output_vector)


def test_get_marginal_probability() -> None:
    state_vector = array(
        [
            0.34371213 - 0.11914814j,
            0.18182189 - 0.26388568j,
            -0.14845723 + 0.16388635j,
            -0.0555532 + 0.0093j,
            0.00761666 - 0.050306j,
            -0.05140567 + 0.00558035j,
            -0.02617409 - 0.09270063j,
            0.10164497 - 0.25085273j,
            -0.33544574 - 0.25489871j,
            0.04471962 - 0.14048391j,
            -0.067943 - 0.21851419j,
            -0.08004672 + 0.10738228j,
            -0.04751262 - 0.23808538j,
            -0.0901756 + 0.38016975j,
            0.07018171 + 0.14937865j,
            0.15221164 + 0.30586554j,
        ]
    )
    assert isclose(
        get_marginal_probability(state_vector, {0: 0, 2: 0}), 0.4110944918747837
    )

    assert isclose(
        get_marginal_probability(state_vector, {0: 1, 2: 0}), 0.14554150383824774
    )

    assert isclose(
        get_marginal_probability(state_vector, {0: 0, 2: 1}), 0.09804874582541159
    )

    assert isclose(
        get_marginal_probability(state_vector, {0: 1, 2: 1}), 0.34531525846155714
    )

    # incorrect input state shape.
    with pytest.raises(
        AssertionError, match="Length of the state vector must be a power of 2."
    ):
        get_marginal_probability(array([1.0, 0.0, 0.0]), {0: 0}),

    # incorrect measured qubit index.
    with pytest.raises(
        AssertionError, match="The specified qubit index 2 is out of range."
    ):
        get_marginal_probability(array([1.0, 0.0, 0.0, 0.0]), {0: 0, 2: 1}),
