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

import stim
from numpy import cfloat, zeros
from numpy.typing import NDArray

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.core.state import GeneralCircuitQuantumState, QuantumStateVector

from .circuit import convert_circuit


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
    state vector.

    Note that in Stim, the output vector gets canonicalized in the way
    that the first non-zero component of the output vector is positive.
    """
    tableau = stim.TableauSimulator()
    tableau.set_state_from_state_vector(init_state, endian="little")
    stim_circuit = convert_circuit(circuit)
    tableau.do_circuit(stim_circuit)
    out_state: NDArray[cfloat] = tableau.state_vector(endian="little")

    return out_state
