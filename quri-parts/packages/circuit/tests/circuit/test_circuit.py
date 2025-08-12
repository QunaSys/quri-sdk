# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import pytest

from quri_parts.circuit import (
    CNOT,
    RX,
    ImmutableQuantumCircuit,
    ParametricRX,
    QuantumCircuit,
    QuantumGate,
    X,
)

_GATES: list[QuantumGate] = [X(0), RX(0, angle=1.0), CNOT(0, 1)]


def mutable_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    for gate in _GATES:
        circuit.add_gate(gate)
    return circuit


def immutable_circuit() -> ImmutableQuantumCircuit:
    q_circuit = mutable_circuit()
    return ImmutableQuantumCircuit(q_circuit)


class TestQuantumCircuit:
    def test_add_gate(self) -> None:
        circuit = mutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit.gates) == len(_GATES)
        assert circuit.gates == tuple(_GATES)
        with pytest.raises((ValueError, TypeError)):
            circuit.add_gate(ParametricRX(0))  # type: ignore
        with pytest.raises(ValueError):
            circuit.add_gate(X(3))

        circuit.add_gate(X(0), 0)
        assert circuit.gates == (X(0),) + tuple(_GATES)
        assert len(circuit.gates) == len(_GATES) + 1

    def test_get_mutable_copy(self) -> None:
        circuit = mutable_circuit()
        circuit_copied = circuit.get_mutable_copy()
        assert id(circuit) != id(circuit_copied)
        assert circuit == circuit_copied

    def test_freeze(self) -> None:
        circuit = mutable_circuit()
        immutable_circuit = circuit.freeze()
        assert isinstance(immutable_circuit, ImmutableQuantumCircuit)
        assert circuit == immutable_circuit
        assert circuit.gates == immutable_circuit.gates

        mut_circuit = immutable_circuit.get_mutable_copy()
        assert circuit == mut_circuit
        assert id(circuit) != id(mut_circuit)
        assert circuit.gates == mut_circuit.gates

    def test_depth(self) -> None:
        circuit = mutable_circuit()
        assert circuit.depth == 3
        immutable_circuit = circuit.freeze()
        assert immutable_circuit.depth == 3

        circuit_2 = QuantumCircuit(3)
        circuit_2.add_X_gate(0)
        circuit_2.add_CNOT_gate(0, 2)
        circuit_2.add_X_gate(1)
        assert circuit_2.depth == 2
        immutable_circuit_2 = circuit_2.freeze()
        assert immutable_circuit_2.depth == 2

    def test_measurement(self) -> None:
        circuit = QuantumCircuit(1, cbit_count=1)
        circuit.add_X_gate(0)
        circuit.measure(0, 0)

        with pytest.raises(ValueError):
            circuit.measure(0, 1)
        with pytest.raises(ValueError):
            circuit.measure(1, 0)

    def test_add(self) -> None:
        circuit1 = mutable_circuit()
        circuit2 = QuantumCircuit(2)
        circuit2.add_X_gate(0)
        combined_circuit = circuit1 + circuit2
        exp_circuit = circuit1.get_mutable_copy()
        exp_circuit.add_X_gate(0)
        assert isinstance(combined_circuit, QuantumCircuit)
        assert combined_circuit.gates == exp_circuit.gates

    def test_compare_pauli_rot(self) -> None:
        circuit1 = QuantumCircuit(3)
        circuit2 = QuantumCircuit(3)
        circuit1.add_H_gate(0)
        circuit1.add_PauliRotation_gate([1, 2], [1, 1], 2)
        circuit1.add_PauliRotation_gate([0, 1, 2], [3, 1, 1], -2)
        circuit1.add_H_gate(0)
        circuit2.add_H_gate(0)
        circuit2.add_PauliRotation_gate([1, 2], [1, 1], 2)
        circuit2.add_PauliRotation_gate([1, 0, 2], [1, 3, 1], -2)
        circuit2.add_H_gate(0)
        assert circuit1 == circuit2

    def test_sample(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        samples = circuit.sample(1000)
        assert len(samples) == 2
        assert sum(samples.values()) == 1000


class TestQuantumCircuitDeprecation:
    def test_order_flip(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            circuit = QuantumCircuit(1, _GATES)  # type: ignore
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "QuantumCircuit initialization takes" in str(w[-1].message)

            no_warning = QuantumCircuit(1, gates=_GATES)
            # Incorrect order constructs identical circuit as correct order.
            assert circuit == no_warning


class TestImmutableQuantumCircuit:
    def test_get_mutable_copy(self) -> None:
        circuit = immutable_circuit()
        mut_circuit = circuit.get_mutable_copy()
        assert isinstance(mut_circuit, QuantumCircuit)
        assert circuit == mut_circuit
        assert id(circuit) != id(mut_circuit)

    def test_freeze(self) -> None:
        circuit = immutable_circuit()
        circuit_ = circuit.freeze()
        assert id(circuit) == id(circuit_)
