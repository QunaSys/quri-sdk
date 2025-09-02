# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import prod
from typing import cast

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit, QuantumGate
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState


def prepare_circuit_hilbert_schmidt_test(
    target_circuit: NonParametricQuantumCircuit,
    trial_circuit: NonParametricQuantumCircuit,
) -> GeneralCircuitQuantumState:
    lattice_size = trial_circuit.qubit_count
    qubit_count = 2 * lattice_size
    trial_circuit = complex_conjugate_circuit(
        expand_circuit(trial_circuit, lattice_size, qubit_count)
    )
    target_circuit = expand_circuit(target_circuit, 0, qubit_count)

    combined_circuit = QuantumCircuit(qubit_count)
    for i in range(lattice_size):
        combined_circuit.add_H_gate(i)
        combined_circuit.add_CNOT_gate(i, i + lattice_size)
    combined_circuit += trial_circuit + target_circuit

    combined_circuit_state = GeneralCircuitQuantumState(qubit_count, combined_circuit)

    return combined_circuit_state


def get_local_hs_operator(index: int, lattice_size: int) -> Operator:
    return Operator(
        {
            PAULI_IDENTITY: 1 / 4,
            pauli_label(f"X {index} X {index + lattice_size}"): 1 / 4,
            pauli_label(f"Y {index} Y {index + lattice_size}"): -1 / 4,
            pauli_label(f"Z {index} Z {index + lattice_size}"): 1 / 4,
        }
    )


def get_hs_operator(alpha: float, lattice_size: int) -> Operator:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the interval (0.0, 1.0)")

    HSfactors = []
    for index in range(lattice_size):
        HSfactors.append(get_local_hs_operator(index, lattice_size))

    combined_operator = Operator()
    if alpha:
        HSoperator = cast(Operator, prod(HSfactors))  # type: ignore[arg-type]
        HSoperator *= -1.0
        HSoperator.add_term(PAULI_IDENTITY, 1.0)
        HSoperator *= alpha
        combined_operator += HSoperator
    if alpha != 1.0:
        LHSoperator = cast(Operator, sum(HSfactors, start=Operator()))
        LHSoperator *= -1.0 / lattice_size
        LHSoperator.add_term(PAULI_IDENTITY, 1.0)
        LHSoperator *= 1 - alpha
        combined_operator += LHSoperator

    return combined_operator


def expand_circuit(
    circuit: NonParametricQuantumCircuit,
    shift: int,
    qubit_count: int,
) -> QuantumCircuit:
    """Expand circuit.

    shift:
        New index to shift the first qubit in the circuit to
    qubit_count:
        Total number of qubits to expand the circuit to
    """
    expanded_circuit = QuantumCircuit(qubit_count)
    for g in circuit.gates:
        expanded_circuit.add_gate(
            QuantumGate(
                g.name,
                tuple(i + shift for i in g.target_indices),
                control_indices=tuple(i + shift for i in g.control_indices),
                classical_indices=g.classical_indices,
                pauli_ids=g.pauli_ids,
                params=g.params,
                unitary_matrix=g.unitary_matrix,
            )
        )

    return expanded_circuit


def complex_conjugate_circuit(circuit: NonParametricQuantumCircuit) -> QuantumCircuit:
    """Complex conjugate circuit.

    This does not give the inverse or Hermitian conjugate of a circuit,
    but only complex conjugates each unitary by selectively reversing
    gate rotation angles.
    """
    conjugated_circuit = QuantumCircuit(circuit.qubit_count)
    for g in circuit.gates:
        if g.pauli_ids.count(2) % 2 or g.name == "RY":
            params = g.params
        else:
            params = tuple(-p for p in g.params)
        conjugated_circuit.add_gate(
            QuantumGate(
                g.name,
                g.target_indices,
                control_indices=g.control_indices,
                classical_indices=g.classical_indices,
                pauli_ids=g.pauli_ids,
                params=params,
                unitary_matrix=g.unitary_matrix,
            )
        )

    return conjugated_circuit
