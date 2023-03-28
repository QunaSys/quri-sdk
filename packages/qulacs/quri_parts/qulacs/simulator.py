from typing import Union

import qulacs as ql
from numpy import cfloat, zeros
from numpy.typing import NDArray

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.state import GeneralCircuitQuantumState, QuantumStateVector
from quri_parts.qulacs.circuit import convert_circuit


def evaluate_state_to_vector(
    state: Union[GeneralCircuitQuantumState, QuantumStateVector]
) -> QuantumStateVector:
    """Convert GeneralCircuitQuantumState or QuantumStateVector to
    QuantumStateVector that only contains the state vector."""
    n_qubits = state.qubit_count

    if isinstance(state, QuantumStateVector):
        init_state_vector = state.vector
    elif isinstance(state, GeneralCircuitQuantumState):
        init_state_vector = zeros(2**n_qubits, dtype=complex)
        init_state_vector[0] = 1.0
    else:
        raise TypeError(
            "the input state should be either a GeneralCircuitQuantumState\
             or a QuantumStateVector"
        )

    out_state_vector = run_circuit(state.circuit, init_state_vector)

    quri_vector_state = QuantumStateVector(n_qubits=n_qubits, vector=out_state_vector)
    return quri_vector_state


def run_circuit(
    circuit: NonParametricQuantumCircuit,
    init_state: NDArray[cfloat],
) -> NDArray[cfloat]:
    """Act a NonParametricQuantumCircuit onto a state vector and returns a new
    state vector."""

    if len(init_state) != 2**circuit.qubit_count:
        raise ValueError("Inconsistent qubit length between circuit and state")

    qulacs_state = ql.QuantumState(circuit.qubit_count)
    qulacs_state.load(init_state)

    qulacs_cicuit = convert_circuit(circuit)
    qulacs_cicuit.update_quantum_state(qulacs_state)

    new_state_vector: NDArray[cfloat] = qulacs_state.get_vector()

    return new_state_vector
