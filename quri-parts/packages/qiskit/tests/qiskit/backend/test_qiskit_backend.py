# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from collections import Counter
from typing import Optional

import pytest
import qiskit
from pydantic.json import pydantic_encoder
from qiskit import qasm3
from qiskit.providers.basic_provider import BasicProvider

from quri_parts.backend import CompositeSamplingJob
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.qiskit.backend import (
    QiskitSamplingBackend,
    QiskitSamplingJob,
    QiskitSavedDataSamplingJob,
    QiskitSavedDataSamplingResult,
)
from quri_parts.qiskit.circuit import convert_circuit


def circuit_converter(
    _: NonParametricQuantumCircuit, transpiler: Optional[CircuitTranspiler] = None
) -> qiskit.QuantumCircuit:
    circuit = qiskit.QuantumCircuit(4)
    circuit.h(0)
    circuit.id(1)
    circuit.h(2)
    circuit.id(3)
    return circuit


def circuit_transpiler(_: NonParametricQuantumCircuit) -> NonParametricQuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    return circuit


class TestQiskitSamplingBackend:
    def test_sample(self) -> None:
        device = BasicProvider().get_backend("basic_simulator")
        backend = QiskitSamplingBackend(device, circuit_converter)
        job = backend.sample(QuantumCircuit(4), 1000)
        counts = job.result().counts

        assert isinstance(job, QiskitSamplingJob)
        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_default_circuit_converter(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_X_gate(0)
        circuit.add_H_gate(1)
        circuit.add_Z_gate(2)

        device = BasicProvider().get_backend("basic_simulator")
        backend = QiskitSamplingBackend(device)
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        assert isinstance(job, QiskitSamplingJob)
        assert set(counts.keys()) == {0b001, 0b011}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_circuit_transpiler(self) -> None:
        circuit = QuantumCircuit(3)
        device = BasicProvider().get_backend("basic_simulator")
        backend = QiskitSamplingBackend(
            device, circuit_transpiler=circuit_transpiler
        )  # With default circuit_converter.
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        assert isinstance(job, QiskitSamplingJob)
        assert set(counts.keys()) == {0b1001, 0b1011}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

    def test_min_shots(self) -> None:
        device = BasicProvider().get_backend("basic_simulator")

        backend = QiskitSamplingBackend(device, circuit_converter)
        backend._min_shots = 1000
        job = backend.sample(QuantumCircuit(4), 50)
        counts = job.result().counts

        assert isinstance(job, QiskitSamplingJob)
        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000

        backend = QiskitSamplingBackend(
            device, circuit_converter, enable_shots_roundup=False
        )
        backend._min_shots = 1000
        with pytest.raises(ValueError):
            job = backend.sample(QuantumCircuit(4), 50)

    def test_shots_split_evenly(self) -> None:
        device = BasicProvider().get_backend("basic_simulator")

        backend = QiskitSamplingBackend(device, circuit_converter)
        backend._max_shots = 100
        job = backend.sample(QuantumCircuit(4), 200)
        counts = job.result().counts

        assert set(counts.keys()) == {0b0000, 0b0001, 0b0100, 0b0101}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 200
        assert isinstance(job, CompositeSamplingJob)
        assert sorted(sum(j.result().counts.values()) for j in job.jobs) == [100, 100]

    def test_max_shots(self) -> None:
        device = BasicProvider().get_backend("basic_simulator")

        backend = QiskitSamplingBackend(device, circuit_converter)
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

        backend = QiskitSamplingBackend(device, circuit_converter)
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

        backend = QiskitSamplingBackend(
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
            _: NonParametricQuantumCircuit,
            transpiler: Optional[CircuitTranspiler] = None,
        ) -> qiskit.QuantumCircuit:
            circuit = qiskit.QuantumCircuit(3)
            circuit.x(1)
            circuit.h(0)
            circuit.z(2)
            return circuit

        qubit_mapping = {
            0: 1,
            1: 0,
            2: 2,
        }

        device = BasicProvider().get_backend("basic_simulator")
        backend = QiskitSamplingBackend(
            device,
            circuit_converter=circuit_converter,
            qubit_mapping=qubit_mapping,
        )
        job = backend.sample(circuit, 1000)
        counts = job.result().counts

        assert set(counts.keys()) == {0b001, 0b011}
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 1000


def test_saving_mode() -> None:
    device = BasicProvider().get_backend("basic_simulator")
    backend = QiskitSamplingBackend(device, save_data_while_sampling=True)
    backend._max_shots = 1000

    qp_circuit = QuantumCircuit(4)
    qp_circuit.add_H_gate(0)
    qp_circuit.add_H_gate(1)
    qp_circuit.add_H_gate(2)
    qp_circuit.add_H_gate(3)
    qp_circuit.add_RX_gate(0, 0.23)
    qp_circuit.add_RY_gate(1, -0.99)
    qp_circuit.add_RX_gate(2, 0.87)
    qp_circuit.add_RZ_gate(3, -0.16)

    qiskit_transpiled_circuit = qiskit.transpile(
        convert_circuit(qp_circuit).measure_all(False), device
    )
    circuit_qasm = qasm3.dumps(qiskit_transpiled_circuit)

    sampling_job = backend.sample(qp_circuit, 2000)
    measurement_counter = sampling_job.result().counts

    # Check if the circuit string and n_shots are distribited correctly
    measurement_counter_from_memory: Counter[int] = Counter()
    for qasm_str, n_shots, saved_job in backend._saved_data:
        measurement_counter_from_memory += Counter(saved_job.result().counts)

        assert qasm_str == circuit_qasm
        assert n_shots == 1000

    # Check if the saved data counter sums up to the correct counter.
    assert measurement_counter_from_memory == measurement_counter

    # Construct the saved data objects
    raw_data_0 = backend._saved_data[0][2]._qiskit_job.result().get_counts()
    raw_data_1 = backend._saved_data[1][2]._qiskit_job.result().get_counts()

    expected_saved_data_0 = QiskitSavedDataSamplingJob(
        circuit_qasm=circuit_qasm,
        n_shots=1000,
        saved_result=QiskitSavedDataSamplingResult(raw_data=raw_data_0),
    )

    expected_saved_data_1 = QiskitSavedDataSamplingJob(
        circuit_qasm=circuit_qasm,
        n_shots=1000,
        saved_result=QiskitSavedDataSamplingResult(raw_data=raw_data_1),
    )

    # Check the jobs and jobs_json properties.
    expected_saved_data_seq = [
        expected_saved_data_0,
        expected_saved_data_1,
    ]
    assert backend.jobs == expected_saved_data_seq
    expected_json_str = json.dumps(expected_saved_data_seq, default=pydantic_encoder)
    assert backend.jobs_json == expected_json_str
