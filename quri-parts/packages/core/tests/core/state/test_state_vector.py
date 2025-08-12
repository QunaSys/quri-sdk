# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from quri_parts.circuit import H, ImmutableQuantumCircuit, QuantumCircuit, Z
from quri_parts.core.state import QuantumStateVector


def a_state() -> QuantumStateVector:
    circuit = QuantumCircuit(2)
    circuit.add_X_gate(1)
    return QuantumStateVector(2, [1.0, 0, 0, 0], circuit)


class TestQuantumStateVector:
    def test_construction(self) -> None:
        state = QuantumStateVector(2, [1.0, 0, 0, 0])
        assert np.all(state.vector == [1.0, 0, 0, 0])

        with pytest.raises(ValueError):
            QuantumStateVector(2, [1.0, 0, 0])

    def test_construction_with_circuit(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_X_gate(1)

        state = QuantumStateVector(2, [1.0, 0, 0, 0], circuit)
        assert np.all(state.vector == [1.0, 0, 0, 0])
        assert isinstance(state.circuit, ImmutableQuantumCircuit)
        assert state.circuit == circuit

        circuit.add_Y_gate(0)
        assert state.circuit != circuit

    def test_qubit_count(self) -> None:
        state = a_state()
        assert state.qubit_count == 2

    def test_with_gates_applied(self) -> None:
        state = a_state()
        state2 = state.with_gates_applied([H(1), Z(0)])

        assert state.circuit != state2.circuit
        assert len(state2.circuit.gates) == 3
