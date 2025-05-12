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

from quri_parts.circuit import (
    CNOT,
    RX,
    ImmutableBoundParametricQuantumCircuit,
    ImmutableParametricQuantumCircuit,
    ParametricPauliRotation,
    ParametricQuantumCircuit,
    ParametricRX,
    QuantumCircuit,
    QuantumGate,
    X,
)

_GATES: list[QuantumGate] = [
    X(0),
    RX(0, angle=1.0),
    CNOT(0, 1),
]

_PARAMS: list[float] = [1.0, 0.5]


def mutable_circuit() -> ParametricQuantumCircuit:
    circuit = ParametricQuantumCircuit(2)
    for gate in _GATES:
        circuit.add_gate(gate)
    circuit.add_ParametricRX_gate(0)
    circuit.add_ParametricPauliRotation_gate([1], [1])
    return circuit


def immutable_circuit() -> ImmutableParametricQuantumCircuit:
    q_circuit = mutable_circuit()
    return ImmutableParametricQuantumCircuit(q_circuit)


class TestUnboundParametricQuantumCircuit:
    def test_unbound_parametric_quantum_circuit(self) -> None:
        circuit = mutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit._gates) == 5
        assert len(circuit._params) == 2
        assert circuit.has_trivial_parameter_mapping

    def test_get_mutable_copy(self) -> None:
        circuit = mutable_circuit()
        circuit_copied = circuit.get_mutable_copy()
        assert isinstance(circuit_copied, ParametricQuantumCircuit)
        assert id(circuit) != id(circuit_copied)
        assert circuit == circuit_copied

    def test_freeze(self) -> None:
        circuit = mutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(immut_circuit, ImmutableParametricQuantumCircuit)
        assert circuit == immut_circuit

    def test_bind_parameters(self) -> None:
        circuit = mutable_circuit()
        circuit_bound = circuit.bind_parameters(_PARAMS)
        assert isinstance(circuit_bound, ImmutableBoundParametricQuantumCircuit)
        assert all([isinstance(gate, QuantumGate) for gate in circuit_bound.gates])
        assert circuit_bound.gates[3].params[0] == 1.0
        assert circuit_bound.gates[4].params[0] == 0.5

    def test_bind_parameters_by_dict(self) -> None:
        circuit = ParametricQuantumCircuit(2)
        for gate in _GATES:
            circuit.add_gate(gate)
        theta = circuit.add_ParametricRX_gate(0)
        phi = circuit.add_ParametricPauliRotation_gate([1], [1])
        param_vals = np.random.random(2).tolist()
        param_dict = {theta: param_vals[0], phi: param_vals[1]}

        expected_circuit = circuit.bind_parameters(param_vals)
        circuit_bound = circuit.bind_parameters_by_dict(param_dict)
        assert expected_circuit.gates == circuit_bound.gates

    def test_depth(self) -> None:
        circuit = mutable_circuit()
        assert circuit.depth == 4
        immut_circuit = circuit.freeze()
        assert immut_circuit.depth == 4
        circuit_bound = circuit.bind_parameters(_PARAMS)
        assert circuit_bound.depth == 4

        circuit_2 = ParametricQuantumCircuit(3)
        circuit_2.add_ParametricRX_gate(0)
        circuit_2.add_CNOT_gate(0, 2)
        circuit_2.add_ParametricRX_gate(1)
        assert circuit_2.depth == 2
        immutable_circuit_2 = circuit_2.freeze()
        assert immutable_circuit_2.depth == 2

    def test_add(self) -> None:
        circuit = mutable_circuit()
        qc_circuit = QuantumCircuit(2)
        qc_circuit.add_H_gate(0)
        got_circuit = circuit + qc_circuit
        exp_circuit = circuit.get_mutable_copy()
        exp_circuit.add_H_gate(0)
        assert got_circuit.gates == exp_circuit.gates

        circuit = mutable_circuit()
        up_circuit = ParametricQuantumCircuit(2)
        up_circuit.add_ParametricRX_gate(0)
        got_circuit = circuit + up_circuit
        exp_circuit = circuit.get_mutable_copy()
        exp_circuit.add_ParametricRX_gate(0)
        assert got_circuit.gates == exp_circuit.gates

    def test_radd(self) -> None:
        circuit = mutable_circuit()
        qc_circuit = QuantumCircuit(2)
        qc_circuit.add_H_gate(0)
        got_circuit = qc_circuit + circuit
        exp_circuit = ParametricQuantumCircuit(2)
        exp_circuit.add_H_gate(0)
        for gate in _GATES:
            exp_circuit.add_gate(gate)
        exp_circuit.add_ParametricRX_gate(0)
        exp_circuit.add_ParametricPauliRotation_gate([1], [1])
        assert got_circuit.gates == exp_circuit.gates

    def test_add_gate(self) -> None:
        circuit = mutable_circuit()
        assert circuit.gates == _GATES + [
            ParametricRX(0),
            ParametricPauliRotation([1], [1]),
        ]
        circuit.add_gate(X(0), 0)
        assert circuit.gates == [X(0)] + _GATES + [
            ParametricRX(0),
            ParametricPauliRotation([1], [1]),
        ]

    def test_sample(self) -> None:
        circuit = ParametricQuantumCircuit(3)
        circuit.add_ParametricRX_gate(0)
        circuit.add_CNOT_gate(0, 2)
        samples = circuit.sample(1000, [np.pi / 4])
        assert len(samples) == 2
        assert sum(samples.values()) == 1000


class TestImmutableUnboundParametricQuantumCircuit:
    def test_immutable_unbound_parametric_quantum_circuit(self) -> None:
        circuit = immutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit._gates) == 5
        assert len(circuit._params) == 2
        assert circuit.has_trivial_parameter_mapping

    def test_get_mutable_copy(self) -> None:
        circuit = immutable_circuit()
        immut_circuit = circuit.get_mutable_copy()
        assert isinstance(immut_circuit, ParametricQuantumCircuit)
        assert id(circuit) != id(immut_circuit)
        assert circuit == immut_circuit

    def test_freeze(self) -> None:
        circuit = immutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(immut_circuit, ImmutableParametricQuantumCircuit)
        assert circuit == immut_circuit

    def test_bind_parameters(self) -> None:
        circuit = immutable_circuit()
        circuit_bound = circuit.bind_parameters(_PARAMS)
        assert isinstance(circuit_bound, ImmutableBoundParametricQuantumCircuit)
        assert all([isinstance(gate, QuantumGate) for gate in circuit_bound.gates])
        assert circuit_bound.gates[3].params[0] == 1.0
        assert circuit_bound.gates[4].params[0] == 0.5
