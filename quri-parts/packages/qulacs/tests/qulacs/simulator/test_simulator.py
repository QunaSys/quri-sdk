# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections import Counter

import pytest
from numpy import allclose, array, isclose, pi
from scipy.stats import unitary_group

from quri_parts.circuit import QuantumCircuit, X
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    QuantumStateVector,
)
from quri_parts.qulacs.circuit import compile_circuit
from quri_parts.qulacs.simulator import (
    create_concurrent_vector_state_sampler,
    create_qulacs_ideal_vector_state_sampler,
    create_qulacs_vector_state_sampler,
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


def test_create_qulacs_vector_state_sampler() -> None:
    state_vector_sampler = create_qulacs_vector_state_sampler()
    n_shots = 1000

    n_qubits = 2
    for i, j in itertools.product(range(2), repeat=2):
        # vector with only circuit
        vector_state = QuantumStateVector(
            n_qubits,
            circuit=QuantumCircuit(n_qubits, gates=[X(1)] * i + [X(0)] * j),
        )
        vector_sampling_cnt = state_vector_sampler(vector_state, n_shots)
        assert vector_sampling_cnt == Counter({2 * i + j: n_shots})

        # vector-valued state with circuit applied
        vector_valued_state = QuantumStateVector(
            n_qubits,
            vector=array([0, 1, 0, 0]),
            circuit=QuantumCircuit(n_qubits, gates=[X(1)] * i + [X(0)] * j),
        )
        vector_valued_sampling_cnt = state_vector_sampler(vector_valued_state, n_shots)
        assert vector_valued_sampling_cnt == Counter({(2 * i + j) ^ (0b01): n_shots})

        # circuit state
        circuit_state = GeneralCircuitQuantumState(
            n_qubits,
            circuit=QuantumCircuit(n_qubits, gates=[X(1)] * i + [X(0)] * j),
        )
        circuit_sampling_cnt = state_vector_sampler(circuit_state, n_shots)
        assert circuit_sampling_cnt == Counter({2 * i + j: n_shots})


def test_create_concurrent_vector_state_sampler() -> None:
    state_vector_concurrent_sampler = create_concurrent_vector_state_sampler()

    n_shots = [1000, 2000, 3000, 4000]
    n_qubits = 2

    # vector with only circuit
    state_shots_tuples = []
    ans = []
    for ind_shot, (i, j) in enumerate(itertools.product(range(2), repeat=2)):
        vector_state = QuantumStateVector(
            n_qubits,
            circuit=QuantumCircuit(n_qubits, gates=[X(1)] * i + [X(0)] * j),
        )
        state_shots_tuples += [(vector_state, n_shots[ind_shot])]
        ans += [Counter({2 * i + j: n_shots[ind_shot]})]
    vector_sampling_cnts = state_vector_concurrent_sampler(state_shots_tuples)
    assert vector_sampling_cnts == ans

    # vector-valued state with circuit applied
    vector_state_shots_tuples = []
    ans = []
    for ind_shot, (i, j) in enumerate(itertools.product(range(2), repeat=2)):
        vector_valued_state = QuantumStateVector(
            n_qubits,
            vector=array([0, 1, 0, 0]),
            circuit=QuantumCircuit(n_qubits, gates=[X(1)] * i + [X(0)] * j),
        )
        vector_state_shots_tuples += [(vector_valued_state, n_shots[ind_shot])]
        ans += [Counter({(2 * i + j) ^ (0b01): n_shots[ind_shot]})]
    vector_sampling_cnts = state_vector_concurrent_sampler(vector_state_shots_tuples)
    assert vector_sampling_cnts == ans

    # circuit state
    circuit_state_shots_tuples = []
    ans = []
    for ind_shot, (i, j) in enumerate(itertools.product(range(2), repeat=2)):
        circuit_state = GeneralCircuitQuantumState(
            n_qubits,
            circuit=QuantumCircuit(n_qubits, gates=[X(1)] * i + [X(0)] * j),
        )
        circuit_state_shots_tuples += [(circuit_state, n_shots[ind_shot])]
        ans += [Counter({2 * i + j: n_shots[ind_shot]})]
    vector_sampling_cnts = state_vector_concurrent_sampler(circuit_state_shots_tuples)
    assert vector_sampling_cnts == ans


def test_create_qulacs_ideal_vector_state_sampler() -> None:
    n_qubits = 2
    n_shots = 1000

    ideal_sampler = create_qulacs_ideal_vector_state_sampler()

    quantum_circuit = QuantumCircuit(n_qubits)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    expected_cnt = {0: 41.90342, 1: 458.09677, 2: 41.90342, 3: 458.09677}

    # vector state with only circuit
    vector_state = QuantumStateVector(n_qubits, circuit=quantum_circuit)
    vector_cnt = ideal_sampler(vector_state, n_shots)
    for i in range(2**n_qubits):
        assert isclose(expected_cnt[i], vector_cnt[i])
    assert isclose(sum(vector_cnt.values()), 1000)

    # circuit state
    circuit_state = GeneralCircuitQuantumState(n_qubits, circuit=quantum_circuit)
    circuit_cnt = ideal_sampler(circuit_state, n_shots)
    for i in range(2**n_qubits):
        assert isclose(expected_cnt[i], circuit_cnt[i])
    assert isclose(sum(circuit_cnt.values()), 1000)

    # test with vector-valued state
    vec = unitary_group.rvs(2**n_qubits)[:, 0]
    circuit = QuantumCircuit(n_qubits, gates=[X(0)])
    vector_state = QuantumStateVector(n_qubits, vec, circuit)

    vector_cnt = ideal_sampler(vector_state, n_shots)

    expected_cnt = {
        0: (abs(vec[1]) ** 2) * n_shots,
        1: (abs(vec[0]) ** 2) * n_shots,
        2: (abs(vec[3]) ** 2) * n_shots,
        3: (abs(vec[2]) ** 2) * n_shots,
    }
    for i in range(2**n_qubits):
        assert isclose(expected_cnt[i], vector_cnt[i])
    assert isclose(sum(vector_cnt.values()), 1000)
