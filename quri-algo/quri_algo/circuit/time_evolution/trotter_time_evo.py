# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

import numpy as np
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
    Parameter,
    QuantumCircuit,
    inverse_circuit,
)
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    PauliLabel,
    pauli_label,
    trotter_suzuki_decomposition,
)

from quri_algo.circuit.utils.transpile import apply_transpiler
from quri_algo.problem import QubitHamiltonian

from .interface import (
    ControlledTimeEvolutionCircuitFactory,
    TimeEvolutionCircuitFactory,
)


def get_shifted_hamiltonian(hamiltonian: Operator, shift: int) -> Operator:
    hamiltonian_shifted = Operator()
    for op, coef in hamiltonian.items():
        if op == PAULI_IDENTITY:
            hamiltonian_shifted.add_term(PAULI_IDENTITY, coef)
            continue
        idx_id_iterable = [
            (idx + shift, id) for idx, id in zip(*op.index_and_pauli_id_list)
        ]
        hamiltonian_shifted.add_term(pauli_label(idx_id_iterable), coef)

    return hamiltonian_shifted


def _add_single_term_for_trotter_time_evolution(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    t: Parameter,
    op: PauliLabel,
    coeff: complex,
    n_trotter: int,
) -> None:
    if op == PAULI_IDENTITY:
        return
    circuit.add_ParametricPauliRotation_gate(
        *op.index_and_pauli_id_list, {t: 2 * coeff.real / n_trotter}
    )


def _get_trotter_pauli_coefficient_pairs(
    hamiltonian: Operator,
    trotter_order: int,
) -> Iterable[tuple[PauliLabel, complex]]:
    if trotter_order == 1:
        return hamiltonian.items()
    else:
        assert trotter_order % 2 == 0, "Trotter order must be 1 or an even number."
        decomposition = trotter_suzuki_decomposition(
            hamiltonian, 1.0, trotter_order // 2
        )
        return [(op, coeff) for op, coeff in decomposition]


def get_trotter_time_evolution_operator(
    hamiltonian: Operator,
    n_state_qubits: int,
    n_trotter: int = 1,
    trotter_order: int = 1,
) -> ImmutableLinearMappedUnboundParametricQuantumCircuit:
    circuit = LinearMappedUnboundParametricQuantumCircuit(n_state_qubits)
    t = circuit.add_parameter("t")
    op_coeff_pair = _get_trotter_pauli_coefficient_pairs(hamiltonian, trotter_order)
    for _ in range(n_trotter):
        for op, coef in op_coeff_pair:
            _add_single_term_for_trotter_time_evolution(circuit, t, op, coef, n_trotter)
    return circuit.freeze()


class FixedStepTrotterTimeEvolutionCircuitFactory(TimeEvolutionCircuitFactory):
    """Factory for generating a Trotter time evolution circuit with a fixed
    number Trotter steps. circuit based on a qubit Hamiltonian.

    Args:
        qubit_hamiltonian: A :class:`QubitHamiltonian`.
        n_trotter: The total number of Trotter steps.
        trotter_order: The Trotter order. Either 1 or an even number.
        transpiler: The transpiler used to transpile the generated circuit.
    """

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        n_trotter: int,
        trotter_order: int = 1,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.n_trotter = n_trotter
        self.qubit_hamiltonian = qubit_hamiltonian
        self.qubit_count = self.qubit_hamiltonian.n_qubit
        self.trotter_order = trotter_order
        self.transpiler = transpiler
        self._param_evo_circuit = get_trotter_time_evolution_operator(
            self.qubit_hamiltonian.qubit_hamiltonian,
            self.qubit_hamiltonian.n_qubit,
            self.n_trotter,
            self.trotter_order,
        )

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        return self._param_evo_circuit.bind_parameters([evolution_time])


def _add_single_term_for_controlled_time_evolution(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    t: Parameter,
    op: PauliLabel,
    coeff: complex,
    n_trotter: int,
) -> None:
    if op == PAULI_IDENTITY:
        circuit.add_ParametricRZ_gate(0, {t: -coeff.real / n_trotter})

    else:
        circuit.add_ParametricPauliRotation_gate(
            *op.index_and_pauli_id_list, {t: coeff.real / n_trotter}
        )
        extended_op = pauli_label(str(op) + f" Z{0}")
        circuit.add_ParametricPauliRotation_gate(
            *extended_op.index_and_pauli_id_list,
            {t: -coeff.real / n_trotter},
        )


def get_trotter_controlled_time_evolution_operator(
    hamiltonian: Operator,
    n_state_qubits: int,
    n_trotter: int = 1,
    trotter_order: int = 1,
) -> ImmutableLinearMappedUnboundParametricQuantumCircuit:
    r"""This part uses the following formulae:

    .. math::
        P_0 \otimes I + P_1 \otimes e^{i\theta P} &= e^{-i \frac{\theta}{2} Z \otimes P} e^{i \frac{\theta}{2} I \otimes P}\\
        P_0 \otimes I + P_1 \otimes e^{i\theta P} &= \text{PauliRotation}(Z \otimes P, \theta) \text{PauliRotation}(I \otimes P, -\theta)\\
        P_0 \otimes I + P_1 \otimes e^{-i \frac{c}{T} P}
        &= \text{PauliRotation}(Z \otimes P, -\frac{c}{T}) \text{PauliRotation}(I \otimes P, \frac{c}{T})
    """
    circuit = LinearMappedUnboundParametricQuantumCircuit(n_state_qubits + 1)
    t = circuit.add_parameter("t")
    hamiltonian_shifted = get_shifted_hamiltonian(hamiltonian, 1)
    op_coeff_pair = _get_trotter_pauli_coefficient_pairs(
        hamiltonian_shifted, trotter_order
    )

    for _ in range(n_trotter):
        for op, coef in op_coeff_pair:
            _add_single_term_for_controlled_time_evolution(
                circuit, t, op, coef, n_trotter
            )
    return circuit.freeze()


class FixedStepTrotterControlledTimeEvolutionCircuitFactory(
    ControlledTimeEvolutionCircuitFactory
):
    """Factory for generating a Trotter controlled time evolution circuit with
    a fixed number Trotter steps. circuit based on a qubit Hamiltonian.

    Args:
        qubit_hamiltonian: A :class:`QubitHamiltonian`.
        n_trotter: The total number of Trotter steps.
        trotter_order: The Trotter order. Either 1 or an even number.
        transpiler: The transpiler used to transpile the generated circuit.
    """

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        n_trotter: int,
        trotter_order: int = 1,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.n_trotter = n_trotter
        self.qubit_hamiltonian = qubit_hamiltonian
        self.qubit_count = qubit_hamiltonian.n_qubit
        self.transpiler = transpiler
        self.trotter_order = trotter_order
        self._param_evo_circuit = get_trotter_controlled_time_evolution_operator(
            self.qubit_hamiltonian.qubit_hamiltonian,
            self.qubit_hamiltonian.n_qubit,
            self.n_trotter,
            self.trotter_order,
        )

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        return self._param_evo_circuit.bind_parameters([evolution_time])


def get_evolution_trotter_step(
    time_step: float, evolution_time: float
) -> tuple[int, int]:
    """Computes the number of repititions needed to implement a Trotter
    (controlled) time evolution circuit."""
    assert time_step > 0, "time step must be greater than 0."
    step = np.abs(evolution_time) / time_step
    if not np.isclose(np.round(step, 12), int(step)):
        raise ValueError(
            f"Evolution time {evolution_time} is not an integer "
            f"multiple of time step {time_step}."
        )
    return int(np.round(step, 12)), int(np.sign(evolution_time))


class FixedIntervalTrotterTimeEvolutionCircuitFactory(TimeEvolutionCircuitFactory):
    """Factory for generating a fixed time step Trotter time evolution circuit
    based on a qubit Hamiltonian.

    Args:
        qubit_hamiltonian: A :class:`QubitHamiltonian`.
        time_step: The time step in a single Trotter step.
        trotter_order: The Trotter order. Either 1 or an even number.
        transpiler: The transpiler used to transpile the generated circuit.
    """

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time_step: float,
        trotter_order: int = 1,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.qubit_hamiltonian = qubit_hamiltonian
        self.qubit_count = qubit_hamiltonian.n_qubit
        self.time_step = time_step
        self.transpiler = transpiler
        self.trotter_order = trotter_order

        self._single_time_step_circuit = get_trotter_time_evolution_operator(
            qubit_hamiltonian.qubit_hamiltonian,
            qubit_hamiltonian.n_qubit,
            trotter_order=trotter_order,
        ).bind_parameters([time_step])

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        n_repition, sgn = get_evolution_trotter_step(self.time_step, evolution_time)
        circuit = QuantumCircuit(self.qubit_count)
        for _ in range(n_repition):
            circuit.extend(self._single_time_step_circuit)
        return circuit if sgn > 0 else inverse_circuit(circuit)


class FixedIntervalTrotterControlledTimeEvolutionCircuitFactory(
    ControlledTimeEvolutionCircuitFactory
):
    """Factory for generating a fixed time step Trotter controlled time
    evolution circuit based on a qubit Hamiltonian.

    Args:
        qubit_hamiltonian: A :class:`QubitHamiltonian`.
        time_step: The time step in a single Trotter step.
        trotter_order: The Trotter order. Either 1 or an even number.
        transpiler: The transpiler used to transpile the generated circuit.
    """

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time_step: float,
        trotter_order: int = 1,
        *,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.qubit_hamiltonian = qubit_hamiltonian
        self.qubit_count = qubit_hamiltonian.n_qubit
        self.time_step = time_step
        self.transpiler = transpiler
        self.trotter_order = trotter_order

        self._single_time_step_circuit = get_trotter_controlled_time_evolution_operator(
            self.qubit_hamiltonian.qubit_hamiltonian,
            self.qubit_hamiltonian.n_qubit,
            1,
            self.trotter_order,
        ).bind_parameters([time_step])

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        n_repition, sgn = get_evolution_trotter_step(self.time_step, evolution_time)
        circuit = QuantumCircuit(self.qubit_count + 1)
        for _ in range(n_repition):
            circuit.extend(self._single_time_step_circuit)
        return circuit if sgn > 0 else inverse_circuit(circuit)


class TrotterTimeEvolutionCircuitFactory(TimeEvolutionCircuitFactory):
    """Factory for generating Trotter time evolution circuit based on a qubit
    Hamiltonian.

    Note:
        You can only specify either time_step or n_trotter.

    Args:
        qubit_hamiltonian: A :class:`QubitHamiltonian`.
        time_step: The time step in a single Trotter step.
        n_trotter: The total number of Trotter steps.
        trotter_order: The Trotter order. Either 1 or an even number.
        transpiler: The transpiler used to transpile the generated circuit.
    """

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        time_step: float | None = None,
        n_trotter: int | None = None,
        trotter_order: int = 1,
        transpiler: CircuitTranspiler | None = None,
    ):
        self.qubit_hamiltonian = qubit_hamiltonian
        self.qubit_count = qubit_hamiltonian.n_qubit
        self.time_step = time_step
        self.n_trotter = n_trotter
        self.trotter_order = trotter_order
        self.transpiler = transpiler

        assert (time_step is not None) ^ (
            n_trotter is not None
        ), "One and only one of time_step and n_trotter is allowed to be specified."

        self._factory: TimeEvolutionCircuitFactory

        if time_step is not None:
            self._factory = FixedIntervalTrotterTimeEvolutionCircuitFactory(
                qubit_hamiltonian, time_step, trotter_order, transpiler=transpiler
            )
        elif n_trotter is not None:
            self._factory = FixedStepTrotterTimeEvolutionCircuitFactory(
                qubit_hamiltonian, n_trotter, trotter_order, transpiler=transpiler
            )

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        return self._factory(evolution_time)


class TrotterControlledTimeEvolutionCircuitFactory(
    ControlledTimeEvolutionCircuitFactory
):
    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        time_step: float | None = None,
        n_trotter: int | None = None,
        trotter_order: int = 1,
        transpiler: CircuitTranspiler | None = None,
    ):
        """Factory for generating Trotter controlled time evolution circuit
        based on a qubit Hamiltonian.

        Note:
            You can only specify either time_step or n_trotter.

        Args:
            qubit_hamiltonian: A :class:`QubitHamiltonian`.
            time_step: The time step in a single Trotter step.
            n_trotter: The total number of Trotter steps.
            trotter_order: The Trotter order. Either 1 or an even number.
            transpiler: The transpiler used to transpile the generated circuit.
        """
        self.qubit_hamiltonian = qubit_hamiltonian
        self.qubit_count = qubit_hamiltonian.n_qubit
        self.time_step = time_step
        self.n_trotter = n_trotter
        self.trotter_order = trotter_order
        self.transpiler = transpiler

        assert (time_step is not None) ^ (
            n_trotter is not None
        ), "One and only one of time_step and n_trotter is allowed to be specified."

        self._factory: ControlledTimeEvolutionCircuitFactory

        if time_step is not None:
            self._factory = FixedIntervalTrotterControlledTimeEvolutionCircuitFactory(
                qubit_hamiltonian, time_step, trotter_order, transpiler=transpiler
            )
        elif n_trotter is not None:
            self._factory = FixedStepTrotterControlledTimeEvolutionCircuitFactory(
                qubit_hamiltonian, n_trotter, trotter_order, transpiler=transpiler
            )

    @apply_transpiler  # type: ignore
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        return self._factory(evolution_time)
