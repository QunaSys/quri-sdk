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
import qulacs

from quri_parts.circuit import (
    LinearMappedParametricQuantumCircuit,
    ParametricQuantumCircuit,
    QuantumCircuit,
    gates,
)
from quri_parts.qulacs.circuit.compiled_circuit import (
    _QulacsCircuit,
    _QulacsLinearMappedUnboundParametricCircuit,
    _QulacsUnboundParametricCircuit,
    compile_circuit,
    compile_parametric_circuit,
)


def gates_equal(g1: qulacs.QuantumGateBase, g2: qulacs.QuantumGateBase) -> bool:
    def gate_info(
        g: qulacs.QuantumGateBase,
    ) -> tuple[str, list[int], list[int]]:
        return (
            g.get_name(),
            g.get_target_index_list(),
            g.get_control_index_list(),
        )

    return (gate_info(g1) == gate_info(g2)) and cast(
        bool, np.all(g1.get_matrix() == g2.get_matrix())
    )


def test_compile_circuit() -> None:
    circuit = QuantumCircuit(3)
    original_gates = [
        gates.X(1),
        gates.H(2),
        gates.CNOT(0, 2),
        gates.RX(0, 0.125),
    ]
    for g in original_gates:
        circuit.add_gate(g)

    compiled_circuit = compile_circuit(circuit)
    assert isinstance(compiled_circuit, _QulacsCircuit)
    qulacs_circuit = compiled_circuit.qulacs_circuit
    assert qulacs_circuit.get_qubit_count() == 3

    expected_gates = [
        qulacs.gate.X(1),
        qulacs.gate.H(2),
        qulacs.gate.CNOT(0, 2),
        qulacs.gate.RX(0, -0.125),
    ]
    assert qulacs_circuit.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(qulacs_circuit.get_gate(i), expected)


def test_compile_parametric_circuit() -> None:
    circuit = ParametricQuantumCircuit(3)
    circuit.add_X_gate(1)
    circuit.add_ParametricRX_gate(0)
    circuit.add_H_gate(2)
    circuit.add_ParametricRY_gate(1)
    circuit.add_CNOT_gate(0, 2)
    circuit.add_ParametricRZ_gate(2)
    circuit.add_RX_gate(0, 0.125)
    circuit.add_ParametricPauliRotation_gate((0, 1, 2), (1, 2, 3))

    compiled_circuit = compile_parametric_circuit(circuit)
    assert isinstance(compiled_circuit, _QulacsUnboundParametricCircuit)
    qulacs_circuit, param_mapper = (
        compiled_circuit.qulacs_circuit,
        compiled_circuit.param_mapper,
    )
    assert qulacs_circuit.get_qubit_count() == 3
    assert param_mapper((0.1, 0.2, 0.3, 0.4)) == (-0.1, -0.2, -0.3, -0.4)

    expected_gates = [
        qulacs.gate.X(1),
        qulacs.gate.ParametricRX(0, 0.0),
        qulacs.gate.H(2),
        qulacs.gate.ParametricRY(1, 0.0),
        qulacs.gate.CNOT(0, 2),
        qulacs.gate.ParametricRZ(2, 0.0),
        qulacs.gate.RX(0, -0.125),
        qulacs.gate.ParametricPauliRotation([0, 1, 2], [1, 2, 3], 0.0),
    ]
    assert qulacs_circuit.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(qulacs_circuit.get_gate(i), expected)


def test_compile_linear_mapped_parametric_circuit() -> None:
    circuit = LinearMappedParametricQuantumCircuit(3)
    theta, phi = circuit.add_parameters("theta", "phi")
    circuit.add_X_gate(1)
    circuit.add_ParametricRX_gate(0, {theta: 0.5})
    circuit.add_H_gate(2)
    circuit.add_ParametricRY_gate(1, phi)
    circuit.add_CNOT_gate(0, 2)
    circuit.add_ParametricRZ_gate(2, {theta: -0.5})
    circuit.add_RX_gate(0, 0.125)
    circuit.add_ParametricPauliRotation_gate((0, 1, 2), (1, 2, 3), {phi: 0.5})
    compiled_circuit = compile_parametric_circuit(circuit)

    assert isinstance(compiled_circuit, _QulacsLinearMappedUnboundParametricCircuit)
    qulacs_circuit, param_mapper = (
        compiled_circuit.qulacs_circuit,
        compiled_circuit.param_mapper,
    )
    assert qulacs_circuit.get_qubit_count() == 3
    assert param_mapper((2.0, 0.5)) == (-1.0, -0.5, 1.0, -0.25)

    expected_gates = [
        qulacs.gate.X(1),
        qulacs.gate.ParametricRX(0, 0.0),
        qulacs.gate.H(2),
        qulacs.gate.ParametricRY(1, 0.0),
        qulacs.gate.CNOT(0, 2),
        qulacs.gate.ParametricRZ(2, 0.0),
        qulacs.gate.RX(0, -0.125),
        qulacs.gate.ParametricPauliRotation([0, 1, 2], [1, 2, 3], 0.0),
    ]
    assert qulacs_circuit.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(qulacs_circuit.get_gate(i), expected)
