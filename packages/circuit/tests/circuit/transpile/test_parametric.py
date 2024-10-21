# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, cast

import numpy as np

from quri_parts.circuit import (
    LinearMappedParametricQuantumCircuit,
    LinearParameterFunction,
    Parameter,
    ParametricQuantumCircuit,
    ParametricQuantumCircuitProtocol,
    ParametricQuantumGate,
    QuantumGate,
    gates,
)
from quri_parts.circuit.transpile import (
    IdentityEliminationTranspiler,
    ParametricTranspiler,
)


def _gates_close(
    x: Union[QuantumGate, ParametricQuantumGate],
    y: Union[QuantumGate, ParametricQuantumGate],
) -> bool:
    if isinstance(x, ParametricQuantumGate) and isinstance(y, ParametricQuantumGate):
        return x == y
    elif isinstance(x, QuantumGate) and isinstance(y, QuantumGate):
        return (
            x.name == y.name
            and x.target_indices == y.target_indices
            and x.control_indices == y.control_indices
            and np.allclose(x.params, y.params)
            and x.pauli_ids == y.pauli_ids
            and np.allclose(x.unitary_matrix, y.unitary_matrix)
        )
    else:
        return False


def _circuit_close(
    x: ParametricQuantumCircuitProtocol,
    y: ParametricQuantumCircuitProtocol,
) -> bool:
    return len(x.gates) == len(y.gates) and all(
        _gates_close(a, b) for a, b in zip(x.gates, y.gates)
    )


def _non_parametric_circuit() -> ParametricQuantumCircuit:
    circuit = ParametricQuantumCircuit(2)
    circuit.extend(
        [
            gates.H(0),
            gates.Identity(1),
            gates.CNOT(0, 1),
        ]
    )
    return circuit


def _parametric_rotation_circuit() -> ParametricQuantumCircuit:
    circuit = ParametricQuantumCircuit(2)
    circuit.extend(
        [
            gates.H(0),
            gates.Identity(1),
            gates.CNOT(0, 1),
        ]
    )
    circuit.add_ParametricRX_gate(1),
    circuit.add_ParametricRY_gate(0),
    circuit.extend(
        [
            gates.T(1),
            gates.Identity(0),
            gates.CZ(1, 0),
        ]
    )
    circuit.add_ParametricRZ_gate(1),
    circuit.extend(
        [
            gates.RX(0, np.pi / 7.0),
            gates.RY(1, np.pi / 4.0),
            gates.RZ(0, np.pi / 3.0),
        ]
    )
    circuit.add_ParametricRX_gate(1),
    return circuit


class TestParametricTranspile:
    def test_identity(self) -> None:
        tr = ParametricTranspiler(lambda circuit: circuit)
        circuit = _parametric_rotation_circuit()
        transpiled = tr(circuit)
        assert _circuit_close(transpiled, circuit)

    def test_identity_elimination(self) -> None:
        tr = ParametricTranspiler(IdentityEliminationTranspiler())
        circuit = _parametric_rotation_circuit()
        transpiled = tr(circuit)
        expect = ParametricQuantumCircuit(2)
        expect.extend(
            [
                gates.H(0),
                gates.CNOT(0, 1),
            ]
        )
        expect.add_ParametricRX_gate(1),
        expect.add_ParametricRY_gate(0),
        expect.extend(
            [
                gates.T(1),
                gates.CZ(1, 0),
            ]
        )
        expect.add_ParametricRZ_gate(1),
        expect.extend(
            [
                gates.RX(0, np.pi / 7.0),
                gates.RY(1, np.pi / 4.0),
                gates.RZ(0, np.pi / 3.0),
            ]
        )
        expect.add_ParametricRX_gate(1),
        assert _circuit_close(transpiled, expect)

    def test_non_parametric(self) -> None:
        tr = ParametricTranspiler(IdentityEliminationTranspiler())
        circuit = _non_parametric_circuit()
        transpiled = tr(circuit)
        expect = ParametricQuantumCircuit(2)
        expect.extend(
            [
                gates.H(0),
                gates.CNOT(0, 1),
            ]
        )
        assert _circuit_close(transpiled, expect)


def _assert_param_map_equal(
    x: LinearMappedParametricQuantumCircuit,
    y: LinearMappedParametricQuantumCircuit,
) -> None:
    xmap, ymap = x.param_mapping.mapping, y.param_mapping.mapping
    assert len(xmap) == len(ymap)
    for (xp, xv), (yp, yv) in zip(xmap.items(), ymap.items()):
        assert xp.name == yp.name
        assert xv == yv


def _assert_param_map_name_equal(
    x: LinearMappedParametricQuantumCircuit,
    y: LinearMappedParametricQuantumCircuit,
) -> None:
    xmap, ymap = x.param_mapping.mapping, y.param_mapping.mapping
    assert len(xmap) == len(ymap)
    for (xp, xv), (yp, yv) in zip(xmap.items(), ymap.items()):
        assert xp.name == yp.name
        if isinstance(xv, Parameter):
            assert xv == yv
        else:
            for (xvp, xvv), (yvp, yvv) in zip(
                cast(LinearParameterFunction, xv).items(),
                cast(LinearParameterFunction, yv).items(),
            ):
                assert xvp.name == yvp.name
                assert xvv == yvv


def _linear_mapped_rotation_circuit() -> LinearMappedParametricQuantumCircuit:
    circuit = LinearMappedParametricQuantumCircuit(3)
    ps = circuit.add_parameters("ps0", "ps1", "ps2")
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_ParametricRX_gate(0, {ps[0]: 0.1, ps[1]: 0.2})
    circuit.add_Identity_gate(0)
    circuit.add_ParametricRY_gate(1, {ps[1]: 0.3, ps[2]: 0.4})
    circuit.add_Identity_gate(1)
    circuit.add_ParametricRZ_gate(0, {ps[0]: 0.5, ps[1]: 0.6})
    circuit.add_Identity_gate(0)
    circuit.add_ParametricPauliRotation_gate(
        [2, 0, 1], [2, 1, 3], {ps[0]: 0.7, ps[1]: 0.8, ps[2]: 0.9}
    )
    return circuit


class TestLinearMappedParametricTranspile:
    def test_identity(self) -> None:
        tr = ParametricTranspiler(lambda circuit: circuit)
        circuit = _linear_mapped_rotation_circuit()
        transpiled = tr(circuit)
        assert _circuit_close(transpiled, circuit)
        _assert_param_map_equal(transpiled, circuit)

    def test_identity_elimination(self) -> None:
        tr = ParametricTranspiler(IdentityEliminationTranspiler())
        circuit = _linear_mapped_rotation_circuit()
        transpiled = tr(circuit)
        expect = LinearMappedParametricQuantumCircuit(3)
        ps = expect.add_parameters("ps0", "ps1", "ps2")
        expect.add_H_gate(0)
        expect.add_CNOT_gate(0, 1)
        expect.add_ParametricRX_gate(0, {ps[0]: 0.1, ps[1]: 0.2})
        expect.add_ParametricRY_gate(1, {ps[1]: 0.3, ps[2]: 0.4})
        expect.add_ParametricRZ_gate(0, {ps[0]: 0.5, ps[1]: 0.6})
        expect.add_ParametricPauliRotation_gate(
            [2, 0, 1], [2, 1, 3], {ps[0]: 0.7, ps[1]: 0.8, ps[2]: 0.9}
        )
        assert _circuit_close(transpiled, expect)
        _assert_param_map_name_equal(transpiled, expect)

    def test_non_parametric(self) -> None:
        tr = ParametricTranspiler(IdentityEliminationTranspiler())
        circuit = LinearMappedParametricQuantumCircuit(2)
        circuit.extend(_non_parametric_circuit())
        transpiled = tr(circuit)
        expect = LinearMappedParametricQuantumCircuit(2)
        expect.extend(
            [
                gates.H(0),
                gates.CNOT(0, 1),
            ]
        )
        assert _circuit_close(transpiled, expect)
        _assert_param_map_equal(transpiled, expect)
