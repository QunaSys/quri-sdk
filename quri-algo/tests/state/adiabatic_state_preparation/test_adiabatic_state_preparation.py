# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import sqrt

import numpy as np
from numpy.testing import assert_almost_equal
from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState

from quri_algo.circuit.time_evolution.trotter_time_evo import (
    TrotterTimeEvolutionCircuitFactory,
)
from quri_algo.problem.operators.hamiltonian import QubitHamiltonian
from quri_algo.state.adiabatic_state_preparation import (
    AdiabaticTimeEvolutionStateFactory,
)

ham_0 = Operator()
ham_0.add_term(pauli_label("Z0"), 1.0)
ham_1 = Operator()
ham_1.add_term(pauli_label("X0"), 1.0)


def ham(th: float) -> QubitHamiltonian:
    return QubitHamiltonian(1, np.cos(th) * ham_0 + np.sin(th) * ham_1)


def assert_circuits_almost_equal(
    circuit: ImmutableQuantumCircuit, expected_circuit: ImmutableQuantumCircuit
) -> None:
    for g0, g1 in zip(circuit.gates, expected_circuit.gates):
        assert g0.classical_indices == g1.classical_indices
        assert g0.control_indices == g1.control_indices
        assert g0.name == g1.name
        assert g0.pauli_ids == g1.pauli_ids
        assert g0.target_indices == g1.target_indices
        assert g0.unitary_matrix == g1.unitary_matrix
        assert_almost_equal(g0.params[0], g1.params[0])


def test_const() -> None:
    DISCRETIZATION = 10
    N_TROTTER = 1
    TROTTER_ORDER = 1
    EVOLUTION_TIME = 10.0

    initial_state = GeneralCircuitQuantumState(1)

    def const_ham(_: float) -> QubitHamiltonian:
        return ham(0.0)

    state_factory = AdiabaticTimeEvolutionStateFactory(
        const_ham, TrotterTimeEvolutionCircuitFactory
    )

    state = state_factory(
        EVOLUTION_TIME,
        DISCRETIZATION,
        initial_state,
        n_trotter=N_TROTTER,
        trotter_order=TROTTER_ORDER,
    )
    trotter_factory = TrotterTimeEvolutionCircuitFactory(
        QubitHamiltonian(1, ham_0),
        n_trotter=DISCRETIZATION - 1,
        trotter_order=TROTTER_ORDER,
    )
    expected_trotter_circuit = trotter_factory(EVOLUTION_TIME)
    assert_circuits_almost_equal(state.circuit, expected_trotter_circuit)

    TROTTER_ORDER = 2

    state = state_factory(
        EVOLUTION_TIME,
        DISCRETIZATION,
        initial_state,
        n_trotter=N_TROTTER,
        trotter_order=TROTTER_ORDER,
    )
    trotter_factory = TrotterTimeEvolutionCircuitFactory(
        QubitHamiltonian(1, ham_0),
        n_trotter=DISCRETIZATION - 1,
        trotter_order=TROTTER_ORDER,
    )
    expected_trotter_circuit = trotter_factory(EVOLUTION_TIME)
    assert_circuits_almost_equal(state.circuit, expected_trotter_circuit)


def test_linear_mapped() -> None:
    DISCRETIZATION = 10
    N_TROTTER = 1
    TROTTER_ORDER = 1
    EVOLUTION_TIME = 10.0

    initial_state = GeneralCircuitQuantumState(1)
    state_factory = AdiabaticTimeEvolutionStateFactory(
        ham, TrotterTimeEvolutionCircuitFactory
    )
    dt = EVOLUTION_TIME / (DISCRETIZATION - 1)

    state = state_factory(
        EVOLUTION_TIME,
        DISCRETIZATION,
        initial_state,
        n_trotter=N_TROTTER,
        trotter_order=TROTTER_ORDER,
    )
    expected_circuit = QuantumCircuit(1)
    for i in range(DISCRETIZATION - 1):
        trotter_factory = TrotterTimeEvolutionCircuitFactory(
            ham((i + 0.5) * dt), n_trotter=N_TROTTER, trotter_order=TROTTER_ORDER
        )
        expected_circuit += trotter_factory(dt)

    assert_circuits_almost_equal(state.circuit, expected_circuit)

    TROTTER_ORDER = 2

    state = state_factory(
        EVOLUTION_TIME,
        DISCRETIZATION,
        initial_state,
        n_trotter=N_TROTTER,
        trotter_order=TROTTER_ORDER,
    )
    expected_circuit = QuantumCircuit(1)
    for i in range(DISCRETIZATION - 1):
        trotter_factory = TrotterTimeEvolutionCircuitFactory(
            ham((i + 0.5) * dt), n_trotter=N_TROTTER, trotter_order=TROTTER_ORDER
        )
        expected_circuit += trotter_factory(dt)

    assert_circuits_almost_equal(state.circuit, expected_circuit)


def test_interpolating() -> None:
    DISCRETIZATION = 10
    N_TROTTER = 1
    TROTTER_ORDER = 1
    EVOLUTION_TIME = 10.0

    initial_state = GeneralCircuitQuantumState(1)
    state_factory = AdiabaticTimeEvolutionStateFactory(
        ham, TrotterTimeEvolutionCircuitFactory
    )

    def interp_function(x: float) -> float:
        return sqrt(x)

    dt = EVOLUTION_TIME / (DISCRETIZATION - 1)
    times = [
        interp_function(i * dt / EVOLUTION_TIME) * EVOLUTION_TIME
        for i in range(DISCRETIZATION)
    ]
    expected_circuit = QuantumCircuit(1)
    for s0, s1 in zip(times[:-1], times[1:]):
        ds = s1 - s0
        s = (s0 + s1) / 2
        trotter_factory = TrotterTimeEvolutionCircuitFactory(
            ham(s), n_trotter=N_TROTTER, trotter_order=TROTTER_ORDER
        )
        expected_circuit += trotter_factory(ds)

    state = state_factory(
        EVOLUTION_TIME,
        DISCRETIZATION,
        initial_state,
        n_trotter=N_TROTTER,
        trotter_order=TROTTER_ORDER,
        interp_function=interp_function,
    )

    assert_circuits_almost_equal(state.circuit, expected_circuit)


def test_stop_at_iteration() -> None:
    DISCRETIZATION = 10
    N_TROTTER = 1
    TROTTER_ORDER = 1
    EVOLUTION_TIME = 10.0
    STOP_AT = 6

    initial_state = GeneralCircuitQuantumState(1)
    state_factory = AdiabaticTimeEvolutionStateFactory(
        ham, TrotterTimeEvolutionCircuitFactory
    )

    def interp_function(x: float) -> float:
        return sqrt(x)

    dt = EVOLUTION_TIME / (DISCRETIZATION - 1)
    times = [
        interp_function(i * dt / EVOLUTION_TIME) * EVOLUTION_TIME
        for i in range(DISCRETIZATION)
    ]
    expected_circuit = QuantumCircuit(1)
    for s0, s1 in zip(times[:STOP_AT], times[1 : STOP_AT + 1]):  # noqa: E203
        ds = s1 - s0
        s = (s0 + s1) / 2
        trotter_factory = TrotterTimeEvolutionCircuitFactory(
            ham(s), n_trotter=N_TROTTER, trotter_order=TROTTER_ORDER
        )
        expected_circuit += trotter_factory(ds)

    state = state_factory(
        EVOLUTION_TIME,
        DISCRETIZATION,
        initial_state,
        n_trotter=N_TROTTER,
        trotter_order=TROTTER_ORDER,
        interp_function=interp_function,
        stop_at_iteration=STOP_AT,
    )

    assert_circuits_almost_equal(state.circuit, expected_circuit)


def test_stop_at_time() -> None:
    DISCRETIZATION = 10
    N_TROTTER = 1
    TROTTER_ORDER = 1
    EVOLUTION_TIME = 10.0
    STOP_AT = 6.0

    initial_state = GeneralCircuitQuantumState(1)
    state_factory = AdiabaticTimeEvolutionStateFactory(
        ham, TrotterTimeEvolutionCircuitFactory
    )

    def interp_function(x: float) -> float:
        return sqrt(x)

    dt = EVOLUTION_TIME / (DISCRETIZATION - 1)
    times = [
        interp_function(i * dt / EVOLUTION_TIME) * EVOLUTION_TIME
        for i in range(DISCRETIZATION)
    ]
    expected_circuit = QuantumCircuit(1)
    for s0, s1 in zip(times[:-1], times[1:]):
        if s0 > STOP_AT:
            break
        ds = s1 - s0
        s = (s0 + s1) / 2
        trotter_factory = TrotterTimeEvolutionCircuitFactory(
            ham(s), n_trotter=N_TROTTER, trotter_order=TROTTER_ORDER
        )
        expected_circuit += trotter_factory(ds)

    state = state_factory(
        EVOLUTION_TIME,
        DISCRETIZATION,
        initial_state,
        n_trotter=N_TROTTER,
        trotter_order=TROTTER_ORDER,
        interp_function=interp_function,
        stop_at_time=STOP_AT,
    )

    assert_circuits_almost_equal(state.circuit, expected_circuit)
