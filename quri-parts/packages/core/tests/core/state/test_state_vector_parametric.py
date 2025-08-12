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

import numpy as np
import pytest

from quri_parts.circuit import (
    H,
    ImmutableParametricQuantumCircuit,
    ParametricQuantumCircuit,
    Z,
)
from quri_parts.core.state import ParametricQuantumStateVector, QuantumStateVector


def a_state() -> ParametricQuantumStateVector:
    circuit = ParametricQuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_Z_gate(1)
    circuit.add_ParametricRX_gate(0)
    return ParametricQuantumStateVector(2, circuit, [1.0, 0, 0, 0])


class TestParametricQuantumStateVector:
    def test_construction_with_circuit(self) -> None:
        circuit = ParametricQuantumCircuit(2)
        circuit.add_H_gate(0)
        circuit.add_Z_gate(1)
        circuit.add_ParametricRX_gate(0)
        state = ParametricQuantumStateVector(2, circuit, [1.0, 0, 0, 0])
        assert np.all(state.vector == [1.0, 0, 0, 0])
        assert isinstance(state.parametric_circuit, ImmutableParametricQuantumCircuit)
        assert state.parametric_circuit == circuit

        circuit.add_Y_gate(0)
        assert state.parametric_circuit != circuit

        with pytest.raises(ValueError):
            ParametricQuantumStateVector(2, circuit, [1.0, 0, 0])

    def test_qubit_count(self) -> None:
        state = a_state()
        assert state.qubit_count == 2

    def test_with_gates_applied(self) -> None:
        state = a_state()
        state2 = state.with_gates_applied([H(1), Z(0)])

        assert state.parametric_circuit != state2.parametric_circuit
        circuit = cast(ParametricQuantumCircuit, state2.parametric_circuit)
        assert len(circuit._gates) == 5

    def test_bind_parameters(self) -> None:
        state = a_state()
        bound_state = state.bind_parameters([0.1])
        assert isinstance(bound_state, QuantumStateVector)
        assert bound_state.qubit_count == 2
        assert np.all(bound_state.vector == state.vector)
        assert len(bound_state.circuit.gates) == 3
