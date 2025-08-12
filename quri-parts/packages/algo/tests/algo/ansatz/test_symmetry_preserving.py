# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence

import numpy as np

from quri_parts.algo.ansatz import SymmetryPreserving, SymmetryPreservingReal
from quri_parts.circuit import (
    CNOT,
    RY,
    RZ,
    ImmutableLinearMappedParametricQuantumCircuit,
    QuantumCircuit,
)


def _build_circuit_qc4_reps3(params_list: Sequence[float]) -> QuantumCircuit:
    gates = []

    theta, phi = params_list[0:2]
    gates.extend(
        [
            CNOT(2, 1),
            RZ(2, -phi - np.pi),
            RY(2, -theta - (0.5 * np.pi)),
            CNOT(1, 2),
            RY(2, theta + (0.5 * np.pi)),
            RZ(2, phi + np.pi),
            CNOT(2, 1),
        ]
    )

    theta, phi = params_list[2:4]
    gates.extend(
        [
            CNOT(1, 0),
            RZ(1, -phi - np.pi),
            RY(1, -theta - (0.5 * np.pi)),
            CNOT(0, 1),
            RY(1, theta + (0.5 * np.pi)),
            RZ(1, phi + np.pi),
            CNOT(1, 0),
        ]
    )

    theta, phi = params_list[4:6]
    gates.extend(
        [
            CNOT(3, 2),
            RZ(3, -phi - np.pi),
            RY(3, -theta - (0.5 * np.pi)),
            CNOT(2, 3),
            RY(3, theta + (0.5 * np.pi)),
            RZ(3, phi + np.pi),
            CNOT(3, 2),
        ]
    )

    theta, phi = params_list[6:8]
    gates.extend(
        [
            CNOT(2, 1),
            RZ(2, -phi - np.pi),
            RY(2, -theta - (0.5 * np.pi)),
            CNOT(1, 2),
            RY(2, theta + (0.5 * np.pi)),
            RZ(2, phi + np.pi),
            CNOT(2, 1),
        ]
    )

    theta, phi = params_list[8:10]
    gates.extend(
        [
            CNOT(1, 0),
            RZ(1, -phi - np.pi),
            RY(1, -theta - (0.5 * np.pi)),
            CNOT(0, 1),
            RY(1, theta + (0.5 * np.pi)),
            RZ(1, phi + np.pi),
            CNOT(1, 0),
        ]
    )

    theta, phi = params_list[10:12]
    gates.extend(
        [
            CNOT(3, 2),
            RZ(3, -phi - np.pi),
            RY(3, -theta - (0.5 * np.pi)),
            CNOT(2, 3),
            RY(3, theta + (0.5 * np.pi)),
            RZ(3, phi + np.pi),
            CNOT(3, 2),
        ]
    )

    theta, phi = params_list[12:14]
    gates.extend(
        [
            CNOT(2, 1),
            RZ(2, -phi - np.pi),
            RY(2, -theta - (0.5 * np.pi)),
            CNOT(1, 2),
            RY(2, theta + (0.5 * np.pi)),
            RZ(2, phi + np.pi),
            CNOT(2, 1),
        ]
    )

    theta, phi = params_list[14:16]
    gates.extend(
        [
            CNOT(1, 0),
            RZ(1, -phi - np.pi),
            RY(1, -theta - (0.5 * np.pi)),
            CNOT(0, 1),
            RY(1, theta + (0.5 * np.pi)),
            RZ(1, phi + np.pi),
            CNOT(1, 0),
        ]
    )

    theta, phi = params_list[16:18]
    gates.extend(
        [
            CNOT(3, 2),
            RZ(3, -phi - np.pi),
            RY(3, -theta - (0.5 * np.pi)),
            CNOT(2, 3),
            RY(3, theta + (0.5 * np.pi)),
            RZ(3, phi + np.pi),
            CNOT(3, 2),
        ]
    )

    circuit = QuantumCircuit(4)

    for gate in gates:
        circuit.add_gate(gate)

    return circuit


def _build_circuit_qc4_reps3_real(params_list: Sequence[float]) -> QuantumCircuit:
    gates = []

    theta = params_list[0]
    gates.extend([CNOT(1, 2), RY(1, theta), CNOT(2, 1), RY(1, -theta), CNOT(1, 2)])

    theta = params_list[1]
    gates.extend([CNOT(0, 1), RY(0, theta), CNOT(1, 0), RY(0, -theta), CNOT(0, 1)])

    theta = params_list[2]
    gates.extend([CNOT(2, 3), RY(2, theta), CNOT(3, 2), RY(2, -theta), CNOT(2, 3)])

    theta = params_list[3]
    gates.extend([CNOT(1, 2), RY(1, theta), CNOT(2, 1), RY(1, -theta), CNOT(1, 2)])

    theta = params_list[4]
    gates.extend([CNOT(0, 1), RY(0, theta), CNOT(1, 0), RY(0, -theta), CNOT(0, 1)])

    theta = params_list[5]
    gates.extend([CNOT(2, 3), RY(2, theta), CNOT(3, 2), RY(2, -theta), CNOT(2, 3)])

    theta = params_list[6]
    gates.extend([CNOT(1, 2), RY(1, theta), CNOT(2, 1), RY(1, -theta), CNOT(1, 2)])

    theta = params_list[7]
    gates.extend([CNOT(0, 1), RY(0, theta), CNOT(1, 0), RY(0, -theta), CNOT(0, 1)])

    theta = params_list[8]
    gates.extend([CNOT(2, 3), RY(2, theta), CNOT(3, 2), RY(2, -theta), CNOT(2, 3)])

    circuit = QuantumCircuit(4)

    for gate in gates:
        circuit.add_gate(gate)

    return circuit


def _test_circuit(
    ansatz: ImmutableLinearMappedParametricQuantumCircuit,
    test_circuit_builder: Callable[[Sequence[float]], QuantumCircuit],
) -> None:
    params_list = list(2.0 * np.pi * np.random.random(ansatz.parameter_count))
    circuit = ansatz.bind_parameters(params_list)
    test_circuit = test_circuit_builder(params_list)
    assert circuit.gates == test_circuit.gates


def test_symmetry_preserving() -> None:
    qubit_count = 4
    reps = 3
    ansatz = SymmetryPreserving(qubit_count, reps)
    assert ansatz.parameter_count == 2 * reps * (qubit_count - 1)
    _test_circuit(ansatz, _build_circuit_qc4_reps3)


def test_symmetry_preserving_real() -> None:
    qubit_count = 4
    reps = 3
    ansatz = SymmetryPreservingReal(qubit_count, reps)
    assert ansatz.parameter_count == reps * (qubit_count - 1)
    _test_circuit(ansatz, _build_circuit_qc4_reps3_real)
