# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.tasks import GateModelQuantumTaskResult

from quri_parts.backend import CompositeSamplingJob
from quri_parts.braket.backend import BraketSamplingBackend, BraketSamplingResult
from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler


def circuit_converter(
    _: ImmutableQuantumCircuit, transpiler: Optional[CircuitTranspiler] = None
) -> Circuit:
    circuit = Circuit()
    circuit.h(0)
    circuit.i(1)
    circuit.h(2)
    circuit.i(3)
    return circuit


def circuit_transpiler(_: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    return circuit


class TestBraketSamplingResult:
    def test_measured_qubits(self) -> None:
        measurements = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
            ]
        )
        braket_result = GateModelQuantumTaskResult(
            task_metadata={},
            additional_metadata={},
            measurements=measurements,
            measured_qubits=[1, 0],
        )
        result = BraketSamplingResult(braket_result)
        counts = result.counts
        assert counts == {0b00: 2, 0b10: 2}
        assert all(isinstance(k, int) for k in counts.keys())

    def test_large_qubits(self) -> None:
        measurements = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 1],
            ]
        )
        braket_result = GateModelQuantumTaskResult(
            task_metadata={},
            additional_metadata={},
            measurements=measurements,
            measured_qubits=[1, 128],
        )
        result = BraketSamplingResult(braket_result)
        counts = result.counts
        assert counts == {0b00: 2, 0b10: 1, 2**128 + 0b10: 1}


class TestBraketSamplingBackend:
    def test_sample(self) -> None:
        device = LocalSimulator()
        backend = BraketSamplingBackend(device, circuit_converter)
        job = backend.sample(QuantumCircuit(4), 1000)
        counts = job.result().counts

        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_default_circuit_converter(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_X_gate(0)
        circuit.add_H_gate(1)
        circuit.add_Z_gate(2)

        device = LocalSimulator()
        backend = BraketSamplingBackend(device)
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        assert set(counts.keys()) == {0b001, 0b011}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_circuit_transpiler(self) -> None:
        circuit = QuantumCircuit(3)

        device = LocalSimulator()
        backend = BraketSamplingBackend(
            device=device, circuit_transpiler=circuit_transpiler
        )  # With default circuit_converter.
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        assert set(counts.keys()) == {0b1001, 0b1011}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_min_shots(self) -> None:
        device = LocalSimulator()

        backend = BraketSamplingBackend(device, circuit_converter)
        backend._min_shots = 1000
        job = backend.sample(QuantumCircuit(4), 50)
        counts = job.result().counts

        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

        backend = BraketSamplingBackend(
            device, circuit_converter, enable_shots_roundup=False
        )
        backend._min_shots = 1000
        with pytest.raises(ValueError):
            job = backend.sample(QuantumCircuit(4), 50)

    def test_shots_split_evenly(self) -> None:
        device = LocalSimulator()

        backend = BraketSamplingBackend(device, circuit_converter)
        backend._max_shots = 100
        job = backend.sample(QuantumCircuit(4), 200)
        counts = job.result().counts

        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 200
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [100, 100]

    def test_max_shots(self) -> None:
        device = LocalSimulator()

        backend = BraketSamplingBackend(device, circuit_converter)
        backend._max_shots = 1000
        job = backend.sample(QuantumCircuit(4), 2100)
        counts = job.result().counts

        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 2100
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [
            100,
            1000,
            1000,
        ]

        backend = BraketSamplingBackend(device, circuit_converter)
        backend._max_shots = 1000
        backend._min_shots = 200
        job = backend.sample(QuantumCircuit(4), 2100)
        counts = job.result().counts

        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 2200
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [
            200,
            1000,
            1000,
        ]

        backend = BraketSamplingBackend(
            device, circuit_converter, enable_shots_roundup=False
        )
        backend._max_shots = 1000
        backend._min_shots = 200
        job = backend.sample(QuantumCircuit(4), 2100)
        counts = job.result().counts

        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 2000
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [1000, 1000]

    def test_qubit_mapping(self) -> None:
        # Original circuit
        circuit = QuantumCircuit(3)
        circuit.add_X_gate(0)
        circuit.add_H_gate(1)
        circuit.add_Z_gate(2)

        def circuit_converter(
            _: ImmutableQuantumCircuit,
            transpiler: Optional[CircuitTranspiler] = None,
        ) -> Circuit:
            circuit = Circuit()
            circuit.x(1)
            circuit.h(0)
            circuit.z(2)
            return circuit

        qubit_mapping = {
            0: 1,
            1: 0,
            2: 2,
        }

        device = LocalSimulator()
        backend = BraketSamplingBackend(
            device,
            circuit_converter=circuit_converter,
            qubit_mapping=qubit_mapping,
        )
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        assert set(counts.keys()) == {0b001, 0b011}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000
