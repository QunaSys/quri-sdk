# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

from quri_parts.circuit import (
    CNOT,
    RX,
    ImmutableBoundParametricQuantumCircuit,
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
    QuantumGate,
    X,
)

_GATES: list[QuantumGate] = [
    X(0),
    RX(0, angle=1.0),
    CNOT(0, 1),
]

_PARAM_VALS: list[float] = [1.0, 0.5]


def mutable_circuit() -> tuple[
    LinearMappedUnboundParametricQuantumCircuit, Sequence[Parameter]
]:
    circuit = LinearMappedUnboundParametricQuantumCircuit(2)
    for gate in _GATES:
        circuit.add_gate(gate)
    params = circuit.add_parameters("RX", "PauliRotation")
    circuit.add_ParametricRX_gate(0, params[0])
    circuit.add_ParametricPauliRotation_gate([1], [1], params[1])
    return circuit, params


def immutable_circuit() -> tuple[
    ImmutableLinearMappedUnboundParametricQuantumCircuit, Sequence[Parameter]
]:
    q_circuit, params = mutable_circuit()
    return ImmutableLinearMappedUnboundParametricQuantumCircuit(q_circuit), params


class TestLinearMappedUnboundParametricQuantumCircuit:
    def test_linear_mapped_unbound_parametric_quantum_circuit(self) -> None:
        circuit, params = mutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit._circuit._gates) == 5
        assert circuit.parameter_count == 2
        assert len(params) == 2
        assert params[0].name == "RX"
        assert params[1].name == "PauliRotation"
        assert circuit.has_trivial_parameter_mapping

    def test_get_mutable_copy(self) -> None:
        circuit, _ = mutable_circuit()
        circuit_copied = circuit.get_mutable_copy()
        assert isinstance(circuit_copied, LinearMappedUnboundParametricQuantumCircuit)
        assert id(circuit) != id(circuit_copied)
        assert circuit._circuit == circuit_copied._circuit

    def test_freeze(self) -> None:
        circuit, _ = mutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(
            immut_circuit, ImmutableLinearMappedUnboundParametricQuantumCircuit
        )

    def test_bind_parameters(self) -> None:
        circuit, _ = mutable_circuit()
        circuit_bound = circuit.bind_parameters(_PARAM_VALS)
        assert isinstance(circuit_bound, ImmutableBoundParametricQuantumCircuit)
        assert all([isinstance(gate, QuantumGate) for gate in circuit_bound.gates])
        assert circuit_bound.gates[3].params[0] == 1.0
        assert circuit_bound.gates[4].params[0] == 0.5

    def test_bind_parameters_linear_mapped(self) -> None:
        circuit = LinearMappedUnboundParametricQuantumCircuit(2)
        for gate in _GATES:
            circuit.add_gate(gate)
        params = circuit.add_parameters("RX", "PauliRotation")
        circuit.add_ParametricRX_gate(
            0,
            {params[0]: 0.0, params[1]: 1.0},
        )
        circuit.add_ParametricPauliRotation_gate(
            [0], [1], {params[0]: 1.0, params[1]: 0.0}
        )
        circuit_bound = circuit.bind_parameters(_PARAM_VALS)
        assert isinstance(circuit_bound, ImmutableBoundParametricQuantumCircuit)
        assert all([isinstance(gate, QuantumGate) for gate in circuit_bound.gates])
        assert circuit_bound.gates[3].params[0] == 0.5
        assert circuit_bound.gates[4].params[0] == 1.0

    def test_depth(self) -> None:
        circuit, _ = mutable_circuit()
        assert circuit.depth == 4
        immut_circuit = circuit.freeze()
        assert immut_circuit.depth == 4
        circuit_bound = circuit.bind_parameters(_PARAM_VALS)
        assert circuit_bound.depth == 4

        circuit_2 = LinearMappedUnboundParametricQuantumCircuit(3)
        params = circuit_2.add_parameters("RX1", "RX2")
        circuit_2.add_ParametricRX_gate(0, params[0])
        circuit_2.add_CNOT_gate(0, 2)
        circuit_2.add_ParametricRX_gate(1, params[1])
        assert circuit_2.depth == 2
        immutable_circuit_2 = circuit_2.freeze()
        assert immutable_circuit_2.depth == 2

    def test_has_trivial_mapping(self) -> None:
        circuit, _ = mutable_circuit()
        params = circuit.add_parameters("RZ")
        circuit.add_ParametricRZ_gate(0, {params[0]: 1.0})
        assert circuit.has_trivial_parameter_mapping

        circuit.add_ParametricRZ_gate(0, {params[0]: 0.5})
        assert not circuit.has_trivial_parameter_mapping

        circuit, _ = mutable_circuit()
        params = circuit.add_parameters("RZ", "RY")
        circuit.add_ParametricRZ_gate(0, {params[0]: 1.0})
        assert not circuit.has_trivial_parameter_mapping

        circuit.add_ParametricRY_gate(1, {params[0]: 1.0, params[1]: 1.0})
        assert not circuit.has_trivial_parameter_mapping


class TestImmutableLinearMappedUnboundParametricQuantumCircuit:
    def test_immutable_linear_mapped_unbound_parametric_quantum_circuit(self) -> None:
        circuit, _ = immutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit._circuit._gates) == 5
        assert circuit.parameter_count == 2
        assert circuit.has_trivial_parameter_mapping

    def test_get_mutable_copy(self) -> None:
        circuit, _ = mutable_circuit()
        circuit_copied = circuit.get_mutable_copy()
        assert isinstance(circuit_copied, LinearMappedUnboundParametricQuantumCircuit)
        assert id(circuit) != id(circuit_copied)
        assert circuit._circuit == circuit_copied._circuit

    def test_freeze(self) -> None:
        circuit, _ = immutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(
            immut_circuit, ImmutableLinearMappedUnboundParametricQuantumCircuit
        )
