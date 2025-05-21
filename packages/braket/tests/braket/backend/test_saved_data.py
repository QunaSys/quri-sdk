# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest
from typing import Any, Optional

import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from quri_parts.backend import CompositeSamplingJob
from quri_parts.braket.backend.saved_sampling import (
    BraketSavedDataSamplingBackend,
    BraketSavedDataSamplingJob,
    BraketSavedDataSamplingResult,
)
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


def get_program_str() -> str:
    circuit = circuit_converter(QuantumCircuit(4))
    return str(circuit.to_ir().json())


class TestSavedSampling(unittest.TestCase):
    json_list: list[dict[str, Any]]
    json_str: str

    @classmethod
    def setUpClass(cls) -> None:
        program_str = get_program_str()
        cls.json_list = [
            {
                "circuit_program_str": program_str,
                "n_shots": 100,
                "saved_result": {
                    "raw_data": {"0000": 50, "1000": 30, "0100": 15, "1100": 5}
                },
            },
            {
                "circuit_program_str": program_str,
                "n_shots": 200,
                "saved_result": {
                    "raw_data": {"0000": 100, "1000": 60, "0100": 30, "1100": 10}
                },
            },
            {
                "circuit_program_str": program_str,
                "n_shots": 200,
                "saved_result": {
                    "raw_data": {"0000": 100, "1000": 60, "0100": 30, "1100": 10}
                },
            },
            {
                "circuit_program_str": program_str,
                "n_shots": 200,
                "saved_result": {
                    "raw_data": {"0000": 100, "1000": 60, "0100": 30, "1100": 10}
                },
            },
            {
                "circuit_program_str": program_str,
                "n_shots": 200,
                "saved_result": {
                    "raw_data": {"0000": 100, "1000": 60, "0100": 30, "1100": 10}
                },
            },
            {
                "circuit_program_str": program_str,
                "n_shots": 200,
                "saved_result": {
                    "raw_data": {"0000": 100, "1000": 60, "0100": 30, "1100": 10}
                },
            },
        ]
        cls.json_str = json.dumps(cls.json_list)

    def test_saved_sampling(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.add_X_gate(0)
        circuit.add_H_gate(1)
        circuit.add_Z_gate(2)

        device = LocalSimulator()

        backend = BraketSavedDataSamplingBackend(
            device,
            self.json_str,
            circuit_converter=circuit_converter,
        )
        backend._max_shots = 200

        saved_job_1 = backend.sample(circuit, 100)
        assert isinstance(saved_job_1, BraketSavedDataSamplingJob)
        assert isinstance(saved_job_1.result(), BraketSavedDataSamplingResult)
        assert saved_job_1.result().counts == {0: 50, 1: 30, 2: 15, 3: 5}

        with pytest.raises(KeyError, match="This experiment is not in the saved data."):
            backend.sample(circuit, 30)

        saved_job_2 = backend.sample(circuit, 1000)
        assert isinstance(saved_job_2, CompositeSamplingJob)
        for job in saved_job_2.jobs:
            assert isinstance(job, BraketSavedDataSamplingJob)
            assert isinstance(job.result(), BraketSavedDataSamplingResult)
        assert saved_job_2.result().counts == {0: 500, 1: 300, 2: 150, 3: 50}

        with pytest.raises(ValueError, match="Replay of this experiment is over"):
            backend.sample(circuit, 300)
