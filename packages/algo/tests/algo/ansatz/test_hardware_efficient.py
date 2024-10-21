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

from quri_parts.algo.ansatz import HardwareEfficient, HardwareEfficientReal
from quri_parts.circuit import (
    CZ,
    RY,
    RZ,
    ImmutableLinearMappedParametricQuantumCircuit,
    QuantumCircuit,
)


def _build_circuit_qc4_d3_sub22(params_list: Sequence[float]) -> QuantumCircuit:
    gates = []

    t0y, t0z, t1y, t1z = params_list[0:4]
    gates.extend([RY(0, t0y), RZ(0, t0z), RY(1, t1y), RZ(1, t1z), CZ(0, 1)])
    t0y, t0z, t1y, t1z = params_list[4:8]
    gates.extend([RY(0, t0y), RZ(0, t0z), RY(1, t1y), RZ(1, t1z), CZ(0, 1)])

    t0y, t0z, t1y, t1z, t2y, t2z, t3y, t3z = params_list[8:16]
    gates.extend(
        [
            RY(0, t0y),
            RZ(0, t0z),
            RY(1, t1y),
            RZ(1, t1z),
            RY(2, t2y),
            RZ(2, t2z),
            RY(3, t3y),
            RZ(3, t3z),
            CZ(1, 2),
            CZ(0, 1),
            CZ(2, 3),
        ]
    )
    t0y, t0z, t1y, t1z, t2y, t2z, t3y, t3z = params_list[16:24]
    gates.extend(
        [
            RY(0, t0y),
            RZ(0, t0z),
            RY(1, t1y),
            RZ(1, t1z),
            RY(2, t2y),
            RZ(2, t2z),
            RY(3, t3y),
            RZ(3, t3z),
            CZ(1, 2),
            CZ(0, 1),
            CZ(2, 3),
        ]
    )
    t0y, t0z, t1y, t1z, t2y, t2z, t3y, t3z = params_list[24:32]
    gates.extend(
        [
            RY(0, t0y),
            RZ(0, t0z),
            RY(1, t1y),
            RZ(1, t1z),
            RY(2, t2y),
            RZ(2, t2z),
            RY(3, t3y),
            RZ(3, t3z),
            CZ(1, 2),
            CZ(0, 1),
            CZ(2, 3),
        ]
    )
    t0y, t0z, t1y, t1z, t2y, t2z, t3y, t3z = params_list[32:40]
    gates.extend(
        [
            RY(0, t0y),
            RZ(0, t0z),
            RY(1, t1y),
            RZ(1, t1z),
            RY(2, t2y),
            RZ(2, t2z),
            RY(3, t3y),
            RZ(3, t3z),
        ]
    )

    circuit = QuantumCircuit(4)

    for gate in gates:
        circuit.add_gate(gate)

    return circuit


def _build_circuit_qc4_d3_sub22_real(params_list: Sequence[float]) -> QuantumCircuit:
    gates = []

    t0, t1 = params_list[0:2]
    gates.extend([RY(0, t0), RY(1, t1), CZ(0, 1)])
    t0, t1 = params_list[2:4]
    gates.extend([RY(0, t0), RY(1, t1), CZ(0, 1)])

    t0, t1, t2, t3 = params_list[4:8]
    gates.extend(
        [
            RY(0, t0),
            RY(1, t1),
            RY(2, t2),
            RY(3, t3),
            CZ(1, 2),
            CZ(0, 1),
            CZ(2, 3),
        ]
    )
    t0, t1, t2, t3 = params_list[8:12]
    gates.extend(
        [
            RY(0, t0),
            RY(1, t1),
            RY(2, t2),
            RY(3, t3),
            CZ(1, 2),
            CZ(0, 1),
            CZ(2, 3),
        ]
    )
    t0, t1, t2, t3 = params_list[12:16]
    gates.extend(
        [
            RY(0, t0),
            RY(1, t1),
            RY(2, t2),
            RY(3, t3),
            CZ(1, 2),
            CZ(0, 1),
            CZ(2, 3),
        ]
    )
    t0, t1, t2, t3 = params_list[16:20]
    gates.extend([RY(0, t0), RY(1, t1), RY(2, t2), RY(3, t3)])

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


def test_hardware_efficient() -> None:
    qubit_count = 4
    reps = 3
    sub_reps = 2
    sub_width = 2
    ansatz = HardwareEfficient(
        qubit_count=qubit_count, reps=reps, sub_width=sub_width, sub_reps=sub_reps
    )
    assert (
        ansatz.parameter_count
        == 2 * qubit_count * (reps + 1) + 2 * sub_width * sub_reps
    )
    _test_circuit(ansatz, _build_circuit_qc4_d3_sub22)


def test_hardware_efficient_real() -> None:
    qubit_count = 4
    reps = 3
    sub_reps = 2
    sub_width = 2
    ansatz = HardwareEfficientReal(
        qubit_count=qubit_count, reps=reps, sub_width=sub_width, sub_reps=sub_reps
    )
    assert ansatz.parameter_count == qubit_count * (reps + 1) + sub_width * sub_reps
    _test_circuit(ansatz, _build_circuit_qc4_d3_sub22_real)
