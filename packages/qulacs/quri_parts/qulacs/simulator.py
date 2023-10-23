# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np
import qulacs as ql
from numpy import cfloat, zeros
from numpy.typing import NDArray

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.state import CircuitQuantumState, QuantumStateVector
from quri_parts.qulacs.circuit import convert_circuit
from quri_parts.qulacs.circuit.compiled_circuit import _QulacsCircuit

from . import cast_to_list


def evaluate_state_to_vector(
    state: Union[CircuitQuantumState, QuantumStateVector]
) -> QuantumStateVector:
    """Convert GeneralCircuitQuantumState or QuantumStateVector to
    QuantumStateVector that only contains the state vector."""
    n_qubits = state.qubit_count

    if isinstance(state, QuantumStateVector):
        init_state_vector = state.vector
    elif isinstance(state, CircuitQuantumState):
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
    qulacs_state.load(cast_to_list(init_state))
    if isinstance(circuit, _QulacsCircuit):
        qulacs_cicuit = circuit._qulacs_circuit
    else:
        qulacs_cicuit = convert_circuit(circuit)

    qulacs_cicuit.update_quantum_state(qulacs_state)

    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    new_state_vector: NDArray[cfloat] = qulacs_state.get_vector()  # type: ignore

    return new_state_vector


def get_marginal_probability(
    state_vector: NDArray[cfloat], measured_values: dict[int, int]
) -> float:
    """Compute the probability of obtaining a result when measuring on a subset
    of the qubits.

    state_vector:
        A 1-dimensional array representing the state vector.
    measured_values:
        A dictionary representing the desired measurement outcome on the specified
        qubtis. Suppose {0: 1, 2: 0} is passed in, it computes the probability of
        obtaining 1 on the 0th qubit and 0 on the 2nd qubit.
    """
    n_qubits: float = np.log2(state_vector.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."
    assert (
        max(measured_values.keys()) < n_qubits
    ), f"The specified qubit index {max(measured_values.keys())} is out of range."

    qulacs_state = ql.QuantumState(int(n_qubits))
    qulacs_state.load(cast_to_list(state_vector))
    measured = [measured_values.get(i, 2) for i in range(int(n_qubits))]
    return qulacs_state.get_marginal_probability(measured)
