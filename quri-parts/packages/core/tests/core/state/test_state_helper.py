# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock, patch

import pytest

from quri_parts.circuit import ParametricQuantumCircuit, QuantumCircuit
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
    apply_circuit,
    quantum_state,
)
from quri_parts.core.state.state_helper import (
    _circuit_quantum_state,
    _quantum_state_vector,
)


def test_circuit_quantum_state() -> None:
    n_qubits = 2
    qc = QuantumCircuit(n_qubits)
    qc.add_H_gate(1)
    state1 = _circuit_quantum_state(n_qubits, qc)
    assert isinstance(state1, GeneralCircuitQuantumState)

    param_qc = ParametricQuantumCircuit(n_qubits)
    param_qc.add_ParametricRX_gate(0)
    state2 = _circuit_quantum_state(n_qubits, param_qc)
    assert isinstance(state2, ParametricCircuitQuantumState)


def test_quantum_state_vector() -> None:
    n_qubits = 2
    state1 = _quantum_state_vector(n_qubits, [1.0, 0, 0, 0], None)
    assert isinstance(state1, QuantumStateVector)

    qc = QuantumCircuit(n_qubits)
    qc.add_H_gate(1)
    state2 = _quantum_state_vector(n_qubits, [1.0, 0, 0, 0], qc)
    assert isinstance(state2, QuantumStateVector)

    param_qc = ParametricQuantumCircuit(n_qubits)
    param_qc.add_ParametricRX_gate(0)
    state3 = _quantum_state_vector(n_qubits, [1.0, 0, 0, 0], param_qc)
    assert isinstance(state3, ParametricQuantumStateVector)


@patch("quri_parts.core.state.state_helper._quantum_state_vector")
@patch("quri_parts.core.state.state_helper._circuit_quantum_state")
def test_quantum_state(
    _circuit_quantum_state_mock: Mock, _quantum_state_vector_mock: Mock
) -> None:
    n_qubits = 2
    _circuit_quantum_state_mock.return_value = GeneralCircuitQuantumState(n_qubits)
    _quantum_state_vector_mock.return_value = QuantumStateVector(n_qubits)

    state1 = quantum_state(n_qubits, bits=0b01)
    assert isinstance(state1, ComputationalBasisState)

    circuit = QuantumCircuit(0)
    state2 = quantum_state(n_qubits, bits=0b01, circuit=circuit)
    assert isinstance(state2, GeneralCircuitQuantumState)
    assert _circuit_quantum_state_mock.call_count == 1

    state3 = quantum_state(n_qubits, vector=[1.0, 0, 0, 0])
    assert isinstance(state3, QuantumStateVector)
    assert _quantum_state_vector_mock.call_count == 1

    with pytest.raises(ValueError):
        quantum_state(n_qubits, vector=[1.0, 0, 0, 0], bits=0b01)  # type: ignore


@patch("quri_parts.core.state.state_helper.quantum_state")
def test_apply_circuit(quantum_state_mock: Mock) -> None:
    n_qubits = 2
    quantum_state_mock.return_value = GeneralCircuitQuantumState(n_qubits)
    state1 = apply_circuit(
        QuantumCircuit(n_qubits), GeneralCircuitQuantumState(n_qubits)
    )
    assert isinstance(state1, GeneralCircuitQuantumState)
    assert quantum_state_mock.call_count == 1

    quantum_state_mock.reset_mock(return_value=True)
    quantum_state_mock.return_value = ParametricCircuitQuantumState(
        n_qubits, ParametricQuantumCircuit(n_qubits)
    )
    state2 = apply_circuit(
        ParametricQuantumCircuit(n_qubits), GeneralCircuitQuantumState(n_qubits)
    )
    assert isinstance(state2, ParametricCircuitQuantumState)
    assert quantum_state_mock.call_count == 1

    quantum_state_mock.reset_mock(return_value=True)
    quantum_state_mock.return_value = QuantumStateVector(n_qubits)
    state3 = apply_circuit(QuantumCircuit(n_qubits), QuantumStateVector(n_qubits))
    assert isinstance(state3, QuantumStateVector)
    assert quantum_state_mock.call_count == 1

    quantum_state_mock.reset_mock(return_value=True)
    quantum_state_mock.return_value = ParametricQuantumStateVector(
        n_qubits, ParametricQuantumCircuit(n_qubits), [1.0, 0, 0, 0]
    )
    state4 = apply_circuit(
        ParametricQuantumCircuit(n_qubits), QuantumStateVector(n_qubits)
    )
    assert isinstance(state4, ParametricQuantumStateVector)
    assert quantum_state_mock.call_count == 1

    quantum_state_mock.reset_mock(return_value=True)
    quantum_state_mock.return_value = ParametricCircuitQuantumState(
        n_qubits, ParametricQuantumCircuit(n_qubits)
    )
    state5 = apply_circuit(
        QuantumCircuit(n_qubits),
        ParametricCircuitQuantumState(n_qubits, ParametricQuantumCircuit(n_qubits)),
    )
    assert isinstance(state5, ParametricCircuitQuantumState)
    assert quantum_state_mock.call_count == 1

    quantum_state_mock.reset_mock(return_value=True)
    quantum_state_mock.return_value = ParametricCircuitQuantumState(
        n_qubits, ParametricQuantumCircuit(n_qubits)
    )
    state6 = apply_circuit(
        ParametricQuantumCircuit(n_qubits),
        ParametricCircuitQuantumState(n_qubits, ParametricQuantumCircuit(n_qubits)),
    )
    assert isinstance(state6, ParametricCircuitQuantumState)
    assert quantum_state_mock.call_count == 1

    quantum_state_mock.reset_mock(return_value=True)
    quantum_state_mock.return_value = ParametricQuantumStateVector(
        n_qubits, ParametricQuantumCircuit(n_qubits), [1.0, 0, 0, 0]
    )
    state7 = apply_circuit(
        QuantumCircuit(n_qubits),
        ParametricQuantumStateVector(
            n_qubits, ParametricQuantumCircuit(n_qubits), [1.0, 0, 0, 0]
        ),
    )
    assert isinstance(state7, ParametricQuantumStateVector)
    assert quantum_state_mock.call_count == 1

    quantum_state_mock.reset_mock(return_value=True)
    quantum_state_mock.return_value = ParametricQuantumStateVector(
        n_qubits, ParametricQuantumCircuit(n_qubits), [1.0, 0, 0, 0]
    )
    state8 = apply_circuit(
        ParametricQuantumCircuit(n_qubits),
        ParametricQuantumStateVector(
            n_qubits, ParametricQuantumCircuit(n_qubits), [1.0, 0, 0, 0]
        ),
    )
    assert isinstance(state8, ParametricQuantumStateVector)
    assert quantum_state_mock.call_count == 1
