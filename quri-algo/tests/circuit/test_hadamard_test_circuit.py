# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import Operator, pauli_label

from quri_algo.circuit.hadamard_test import (
    HadamardTestCircuitFactory,
    construct_hadamard_circuit,
)
from quri_algo.circuit.time_evolution.trotter_time_evo import (
    TrotterControlledTimeEvolutionCircuitFactory,
    get_shifted_hamiltonian,
)
from quri_algo.problem import QubitHamiltonian


def test_get_shifted_hamiltonian() -> None:
    hamiltonian = Operator(
        {pauli_label("X0 X2"): 1, pauli_label("Z1 Y3"): 1, pauli_label("X0 Z2 Y3"): 1}
    )
    assert get_shifted_hamiltonian(hamiltonian, 1) == Operator(
        {pauli_label("X1 X3"): 1, pauli_label("Z2 Y4"): 1, pauli_label("X1 Z3 Y4"): 1}
    )
    assert get_shifted_hamiltonian(hamiltonian, 2) == Operator(
        {pauli_label("X2 X4"): 1, pauli_label("Z3 Y5"): 1, pauli_label("X2 Z4 Y5"): 1}
    )


def test_construct_hadamard_circuit_real() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    h_input = QubitHamiltonian(2, hamiltonian)
    evo_time = 2.0
    n_trotter = 1

    circuit_expected = QuantumCircuit(3)
    circuit_expected.add_H_gate(0)
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time)
    circuit_expected.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -coef * evo_time)
    circuit_expected.add_H_gate(0)

    time_evo_generator = TrotterControlledTimeEvolutionCircuitFactory(
        h_input, n_trotter=n_trotter
    )

    circuit = construct_hadamard_circuit(time_evo_generator(evo_time), True)

    for g1, g2 in zip(circuit_expected.gates, circuit.gates):
        assert g1 == g2


def test_construct_hadamard_circuit_imag() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    h_input = QubitHamiltonian(2, hamiltonian)
    evo_time = 2.0
    n_trotter = 1

    circuit_expected = QuantumCircuit(3)
    circuit_expected.add_H_gate(0)
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time)
    circuit_expected.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -coef * evo_time)
    circuit_expected.add_Sdag_gate(0)
    circuit_expected.add_H_gate(0)

    time_evo_generator = TrotterControlledTimeEvolutionCircuitFactory(
        h_input, n_trotter=n_trotter
    )

    circuit = construct_hadamard_circuit(time_evo_generator(evo_time), False)

    for g1, g2 in zip(circuit_expected.gates, circuit.gates):
        assert g1 == g2


def test_construct_hadamard_circuit_pre() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    h_input = QubitHamiltonian(2, hamiltonian)
    evo_time = 2.0
    n_trotter = 1
    pre = QuantumCircuit(3)
    pre.add_CNOT_gate(0, 1)

    circuit_expected = QuantumCircuit(3)
    circuit_expected.add_H_gate(0)
    circuit_expected.add_CNOT_gate(0, 1)
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time)
    circuit_expected.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -coef * evo_time)
    circuit_expected.add_H_gate(0)

    time_evo_generator = TrotterControlledTimeEvolutionCircuitFactory(
        h_input, n_trotter=n_trotter
    )

    circuit = construct_hadamard_circuit(
        time_evo_generator(evo_time), True, preprocess_circuit=pre
    )

    for g1, g2 in zip(circuit_expected.gates, circuit.gates):
        assert g1 == g2


def test_construct_hadamard_circuit_post() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    h_input = QubitHamiltonian(2, hamiltonian)
    evo_time = 2.0
    n_trotter = 1
    post = QuantumCircuit(3)
    post.add_CNOT_gate(0, 1)

    circuit_expected = QuantumCircuit(3)
    circuit_expected.add_H_gate(0)
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time)
    circuit_expected.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -coef * evo_time)
    circuit_expected.add_CNOT_gate(0, 1)
    circuit_expected.add_H_gate(0)

    time_evo_generator = TrotterControlledTimeEvolutionCircuitFactory(
        h_input, n_trotter=n_trotter
    )

    circuit = construct_hadamard_circuit(
        time_evo_generator(evo_time), True, postprocess_circuit=post
    )

    for g1, g2 in zip(circuit_expected.gates, circuit.gates):
        assert g1 == g2


def test_construct_hadamard_circuit_five_trotter() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    h_input = QubitHamiltonian(2, hamiltonian)
    evo_time = 2.0
    n_trotter = 5
    post = QuantumCircuit(3)
    post.add_CNOT_gate(0, 1)

    circuit_expected = QuantumCircuit(3)
    circuit_expected.add_H_gate(0)
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time / n_trotter)
    circuit_expected.add_PauliRotation_gate(
        [0, 1, 2], [3, 1, 1], -coef * evo_time / n_trotter
    )
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time / n_trotter)
    circuit_expected.add_PauliRotation_gate(
        [0, 1, 2], [3, 1, 1], -coef * evo_time / n_trotter
    )
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time / n_trotter)
    circuit_expected.add_PauliRotation_gate(
        [0, 1, 2], [3, 1, 1], -coef * evo_time / n_trotter
    )
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time / n_trotter)
    circuit_expected.add_PauliRotation_gate(
        [0, 1, 2], [3, 1, 1], -coef * evo_time / n_trotter
    )
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time / n_trotter)
    circuit_expected.add_PauliRotation_gate(
        [0, 1, 2], [3, 1, 1], -coef * evo_time / n_trotter
    )
    circuit_expected.add_CNOT_gate(0, 1)
    circuit_expected.add_H_gate(0)

    time_evo_generator = TrotterControlledTimeEvolutionCircuitFactory(
        h_input, n_trotter=n_trotter
    )

    circuit = construct_hadamard_circuit(
        time_evo_generator(evo_time), True, postprocess_circuit=post
    )

    for g1, g2 in zip(circuit_expected.gates, circuit.gates):
        assert g1 == g2


def test_construct_hadamard_circuit_all() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    h_input = QubitHamiltonian(2, hamiltonian)
    evo_time = 2.0
    n_trotter = 1
    post = QuantumCircuit(3)
    post.add_CNOT_gate(0, 1)
    pre = QuantumCircuit(3)
    pre.add_CZ_gate(0, 1)

    circuit_expected = QuantumCircuit(3)
    circuit_expected.add_H_gate(0)
    circuit_expected.add_CZ_gate(0, 1)
    circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time)
    circuit_expected.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -coef * evo_time)
    circuit_expected.add_CNOT_gate(0, 1)
    circuit_expected.add_Sdag_gate(0)
    circuit_expected.add_H_gate(0)

    time_evo_generator = TrotterControlledTimeEvolutionCircuitFactory(
        h_input, n_trotter=n_trotter
    )

    circuit = construct_hadamard_circuit(
        time_evo_generator(evo_time),
        False,
        preprocess_circuit=pre,
        postprocess_circuit=post,
    )

    for g1, g2 in zip(circuit_expected.gates, circuit.gates):
        assert g1 == g2


def test_hadamard_test_circuit_compiler() -> None:
    coef = 1.0
    hamiltonian = Operator({pauli_label("X 0 X 1"): coef})
    h_input = QubitHamiltonian(2, hamiltonian)
    evo_time = 2.0
    n_trotter = 1

    time_evo_generator = TrotterControlledTimeEvolutionCircuitFactory(
        h_input, n_trotter=n_trotter
    )

    for test_real in [True, False]:
        circuit_expected = QuantumCircuit(3)
        circuit_expected.add_H_gate(0)
        circuit_expected.add_PauliRotation_gate([1, 2], [1, 1], coef * evo_time)
        circuit_expected.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -coef * evo_time)
        if not test_real:
            circuit_expected.add_Sdag_gate(0)
        circuit_expected.add_H_gate(0)

        hadamard_test_circuit_generator = HadamardTestCircuitFactory(
            test_real, time_evo_generator
        )
        hadamard_test_circuit = hadamard_test_circuit_generator(evo_time)

        assert circuit_expected == hadamard_test_circuit
