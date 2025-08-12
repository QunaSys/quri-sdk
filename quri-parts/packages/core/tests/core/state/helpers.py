# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast

from quri_parts.core.state import CircuitQuantumState, StateVectorType


def raw_vector(state: CircuitQuantumState) -> StateVectorType:
    from qulacs import QuantumState as QulacsQuantumState

    from quri_parts.qulacs.circuit import convert_circuit

    circuit = state.circuit
    qs = QulacsQuantumState(state.qubit_count)
    convert_circuit(circuit).update_quantum_state(qs)
    return cast(StateVectorType, qs.get_vector())
