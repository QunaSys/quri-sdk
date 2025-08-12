# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.backend import SamplingCounts
from quri_parts.backend.qubit_mapping import (
    BackendQubitMapping,
    QubitMappedSamplingJob,
    QubitMappedSamplingResult,
)
from quri_parts.circuit.transpile import QubitRemappingTranspiler

qubit_mapping = {
    0: 4,
    1: 2,
    2: 0,
}

original_counts = {
    0b000: 100,
    0b001: 200,
    0b010: 300,
    0b011: 400,
    0b100: 500,
    0b101: 600,
    0b110: 700,
    0b111: 800,
}

mapped_counts = {
    # 0b000
    0b00000: 20,
    0b00010: 10,
    0b01000: 70,
    # 0b001
    0b10000: 50,
    0b11010: 150,
    # 0b010
    0b00100: 300,
    # 0b011
    0b10100: 400,
    # 0b100
    0b00001: 500,
    # 0b101
    0b10001: 600,
    # 0b110
    0b00101: 700,
    # 0b111
    0b10101: 800,
}


class TestBackendQubitMapping:
    def test_circuit_transpiler(self) -> None:
        mapping = BackendQubitMapping(qubit_mapping)
        transpiler = mapping.circuit_transpiler
        assert isinstance(transpiler, QubitRemappingTranspiler)

    def test_unmap_sampling_counts(self) -> None:
        mapping = BackendQubitMapping(qubit_mapping)
        unmapped = mapping.unmap_sampling_counts(mapped_counts)
        assert unmapped == original_counts


class MockSamplingResult:
    @property
    def counts(self) -> SamplingCounts:
        return mapped_counts


def test_qubit_mapped_sampling_result() -> None:
    mapping = BackendQubitMapping(qubit_mapping)
    result = QubitMappedSamplingResult(MockSamplingResult(), mapping)
    assert result.counts == original_counts


class MockSamplingJob:
    def result(self) -> MockSamplingResult:
        return MockSamplingResult()


def test_qubit_mapped_sampling_job() -> None:
    mapping = BackendQubitMapping(qubit_mapping)
    job = QubitMappedSamplingJob(MockSamplingJob(), mapping)
    result = job.result()
    assert result.counts == original_counts
