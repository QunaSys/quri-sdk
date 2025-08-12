# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import H, ImmutableQuantumCircuit, QuantumCircuit, Z
from quri_parts.core.state import GeneralCircuitQuantumState


def a_state() -> GeneralCircuitQuantumState:
    s = GeneralCircuitQuantumState(2)
    s = s.with_gates_applied([H(0), Z(1)])
    return s


class TestGeneralCircuitQuantumState:
    def test_construction_with_circuit(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_X_gate(1)

        state = GeneralCircuitQuantumState(3, circuit)
        assert len(state._circuit.gates) == 1

        circuit.add_Y_gate(2)
        assert len(state._circuit.gates) == 1

    def test_circuit(self) -> None:
        state = a_state()
        circuit = state.circuit
        assert isinstance(circuit, ImmutableQuantumCircuit)
        assert len(circuit.gates) == 2

    def test_qubit_count(self) -> None:
        state = a_state()
        assert state.qubit_count == 2

    def test_with_gates_applied(self) -> None:
        state = a_state()
        state2 = state.with_gates_applied([H(1), Z(0)])

        assert state._circuit != state2._circuit
        assert len(state2._circuit.gates) == 4
