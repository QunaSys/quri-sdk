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

from quri_parts.circuit import (
    H,
    ImmutableParametricQuantumCircuit,
    ParametricQuantumCircuit,
    Z,
)
from quri_parts.core.state import (
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
)


def a_state() -> ParametricCircuitQuantumState:
    circuit = ParametricQuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_Z_gate(1)
    circuit.add_ParametricRX_gate(0)
    s = ParametricCircuitQuantumState(2, circuit)
    return s


class TestParametricCircuitQuantumState:
    def test_parametric_circuit(self) -> None:
        state = a_state()
        circuit = state.parametric_circuit
        assert isinstance(circuit, ImmutableParametricQuantumCircuit)
        assert len(circuit._gates) == 3

    def test_qubit_count(self) -> None:
        state = a_state()
        assert state.qubit_count == 2

    def test_with_gates_applied(self) -> None:
        state = a_state()
        state2 = state.with_gates_applied([H(1), Z(0)])
        assert state._circuit != state2._circuit
        circuit = cast(ParametricQuantumCircuit, state2._circuit)
        assert len(circuit._gates) == 5

    def test_bind_parameters(self) -> None:
        state = a_state()
        bound_state = state.bind_parameters([0.1])
        assert isinstance(bound_state, GeneralCircuitQuantumState)
        assert bound_state.qubit_count == 2
        assert len(bound_state.circuit.gates) == 3
