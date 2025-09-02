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

import numpy as np

from quri_parts.circuit import (
    CNOT,
    RX,
    ImmutableBoundParametricQuantumCircuit,
    ImmutableLinearMappedParametricQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
    Parameter,
    ParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    X,
)

_GATES: list[QuantumGate] = [
    X(0),
    RX(0, angle=1.0),
    CNOT(0, 1),
]

_PARAM_VALS: list[float] = [1.0, 0.5]


def mutable_circuit() -> (
    tuple[LinearMappedParametricQuantumCircuit, Sequence[Parameter]]
):
    circuit = LinearMappedParametricQuantumCircuit(2)
    for gate in _GATES:
        circuit.add_gate(gate)
    params = circuit.add_parameters("RX", "PauliRotation")
    circuit.add_ParametricRX_gate(0, params[0])
    circuit.add_ParametricPauliRotation_gate([1], [1], params[1])
    return circuit, params


def immutable_circuit() -> (
    tuple[ImmutableLinearMappedParametricQuantumCircuit, Sequence[Parameter]]
):
    q_circuit, params = mutable_circuit()
    return ImmutableLinearMappedParametricQuantumCircuit(q_circuit), params


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
        assert isinstance(circuit_copied, LinearMappedParametricQuantumCircuit)
        assert id(circuit) != id(circuit_copied)
        assert circuit._circuit == circuit_copied._circuit

    def test_freeze(self) -> None:
        circuit, _ = mutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(immut_circuit, ImmutableLinearMappedParametricQuantumCircuit)

    def test_bind_parameters(self) -> None:
        circuit, _ = mutable_circuit()
        circuit_bound = circuit.bind_parameters(_PARAM_VALS)
        assert isinstance(circuit_bound, ImmutableBoundParametricQuantumCircuit)
        assert all([isinstance(gate, QuantumGate) for gate in circuit_bound.gates])
        assert circuit_bound.gates[3].params[0] == 1.0
        assert circuit_bound.gates[4].params[0] == 0.5

    def test_bind_parameters_by_dict(self) -> None:
        circuit, params = mutable_circuit()
        rx, pr = params
        param_vals = np.random.random(2).tolist()

        expected_circuit = circuit.bind_parameters(param_vals)
        bound_circuit = circuit.bind_parameters_by_dict(
            {rx: param_vals[0], pr: param_vals[1]}
        )
        assert expected_circuit.gates == bound_circuit.gates

    def test_bind_parameters_by_dict_linear_mapped(self) -> None:
        circuit = LinearMappedParametricQuantumCircuit(2)
        for gate in _GATES:
            circuit.add_gate(gate)
        params = circuit.add_parameters("RX", "PauliRotation")
        coeffs = np.random.random(4).tolist()
        circuit.add_ParametricRX_gate(
            0,
            {params[0]: coeffs[0], params[1]: coeffs[1]},
        )
        circuit.add_ParametricPauliRotation_gate(
            [0], [1], {params[0]: coeffs[2], params[1]: coeffs[3]}
        )

        param_vals = np.random.random(2).tolist()
        expected_circuit = circuit.bind_parameters(param_vals)
        circuit_bound = circuit.bind_parameters_by_dict(
            {params[i]: param_vals[i] for i in range(2)}
        )
        assert expected_circuit.gates == circuit_bound.gates
        print(circuit.param_mapping.mapping)

    def test_bind_parameters_linear_mapped(self) -> None:
        circuit = LinearMappedParametricQuantumCircuit(2)
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

        circuit_2 = LinearMappedParametricQuantumCircuit(3)
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

    def test_add(self) -> None:
        circuit = LinearMappedParametricQuantumCircuit(2)
        params = circuit.add_parameters("RX", "RY")
        circuit.add_ParametricRX_gate(0, params[0])
        circuit.add_ParametricRY_gate(1, params[1])

        lm_circuit = LinearMappedParametricQuantumCircuit(2)
        lm_param = lm_circuit.add_parameter("RZ")
        lm_circuit.add_ParametricRZ_gate(0, lm_param)
        got_circuit = circuit + lm_circuit
        exp_circuit = circuit.get_mutable_copy()
        exp_param = exp_circuit.add_parameter("RZ")
        exp_circuit.add_ParametricRZ_gate(0, exp_param)
        assert got_circuit.gates == exp_circuit.gates
        got_vals = got_circuit.param_mapping.mapper(
            {params[0]: 1.0, params[1]: 0.7, lm_param: 0.4}
        )
        exp_vals = exp_circuit.param_mapping.mapper(
            {params[0]: 1.0, params[1]: 0.7, exp_param: 0.4}
        )
        for got, exp in zip(got_vals.values(), exp_vals.values()):
            assert got == exp

        up_circuit = ParametricQuantumCircuit(2)
        in_param = up_circuit.add_ParametricRZ_gate(0)
        got_circuit = circuit + up_circuit
        exp_circuit = circuit.get_mutable_copy()
        exp_param = exp_circuit.add_parameter("RZ")
        exp_circuit.add_ParametricRZ_gate(0, exp_param)
        assert got_circuit.gates == exp_circuit.gates
        got_vals = got_circuit.param_mapping.mapper(
            {params[0]: 1.0, params[1]: 0.7, in_param: 0.4}
        )
        exp_vals = exp_circuit.param_mapping.mapper(
            {params[0]: 1.0, params[1]: 0.7, exp_param: 0.4}
        )
        for got, exp in zip(got_vals.values(), exp_vals.values()):
            assert got == exp

        qc_circuit = QuantumCircuit(2)
        qc_circuit.add_H_gate(0)
        got_circuit = circuit + qc_circuit
        exp_circuit = circuit.get_mutable_copy()
        exp_circuit.add_H_gate(0)
        assert got_circuit.gates == exp_circuit.gates
        got_vals = got_circuit.param_mapping.mapper({params[0]: 1.0, params[1]: 0.7})
        exp_vals = exp_circuit.param_mapping.mapper({params[0]: 1.0, params[1]: 0.7})
        for got, exp in zip(got_vals.values(), exp_vals.values()):
            assert got == exp

    def test_radd(self) -> None:
        circuit = LinearMappedParametricQuantumCircuit(2)
        params = circuit.add_parameters("RX", "PauliRotation")
        circuit.add_ParametricRX_gate(0, params[0])
        circuit.add_ParametricPauliRotation_gate([1], [1], params[1])

        qc_circuit = QuantumCircuit(2)
        qc_circuit.add_H_gate(0)
        got_circuit = qc_circuit + circuit
        exp_circuit = LinearMappedParametricQuantumCircuit(2)
        exp_circuit.add_H_gate(0)
        exp_params = exp_circuit.add_parameters("RX", "PauliRotation")
        exp_circuit.add_ParametricRX_gate(0, exp_params[0])
        exp_circuit.add_ParametricPauliRotation_gate([1], [1], exp_params[1])
        assert got_circuit.gates == exp_circuit.gates
        got_vals = got_circuit.param_mapping.mapper({params[0]: 1.0, params[1]: 0.8})
        exp_vals = exp_circuit.param_mapping.mapper(
            {exp_params[0]: 1.0, exp_params[1]: 0.8}
        )
        for got, exp in zip(got_vals.values(), exp_vals.values()):
            assert got == exp

        up_circuit = ParametricQuantumCircuit(2)
        up_param = up_circuit.add_ParametricRZ_gate(0)
        got_circuit2 = up_circuit + circuit
        exp_circuit = LinearMappedParametricQuantumCircuit(2)
        exp_params = exp_circuit.add_parameters("RZ", "RX", "PauliRotation")
        exp_circuit.add_ParametricRZ_gate(0, exp_params[0])
        exp_circuit.add_ParametricRX_gate(0, exp_params[1])
        exp_circuit.add_ParametricPauliRotation_gate([1], [1], exp_params[2])
        assert got_circuit2.gates == exp_circuit.gates
        got_vals = got_circuit2.param_mapping.mapper(
            {up_param: 0.4, params[0]: 1.0, params[1]: 0.8}
        )
        exp_vals = exp_circuit.param_mapping.mapper(
            {exp_params[0]: 0.4, exp_params[1]: 1.0, exp_params[2]: 0.8}
        )
        for got, exp in zip(got_vals.values(), exp_vals.values()):
            assert got == exp


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
        assert isinstance(circuit_copied, LinearMappedParametricQuantumCircuit)
        assert id(circuit) != id(circuit_copied)
        assert circuit._circuit == circuit_copied._circuit

    def test_freeze(self) -> None:
        circuit, _ = immutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(immut_circuit, ImmutableLinearMappedParametricQuantumCircuit)
