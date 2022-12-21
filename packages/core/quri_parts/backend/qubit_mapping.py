# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from quri_parts.circuit.transpile import QubitRemappingTranspiler

from . import SamplingCounts, SamplingJob, SamplingResult


def _create_reverse_map(qubit_mapping: Mapping[int, int]) -> Mapping[int, int]:
    return {1 << v: 1 << k for k, v in qubit_mapping.items()}


def _reverse_map_bits(x: int, reverse_bit_map: Mapping[int, int]) -> int:
    result = 0
    for mapped_bits, original_bits in reverse_bit_map.items():
        if (x & mapped_bits) != 0:
            result += original_bits
    return result


def _reverse_map_counts(
    m: SamplingCounts, reverse_bit_map: Mapping[int, int]
) -> SamplingCounts:
    d: dict[int, int] = {}
    for b, count in m.items():
        reversed_bits = _reverse_map_bits(b, reverse_bit_map)
        d[reversed_bits] = d.get(reversed_bits, 0) + count
    return d


@dataclass(frozen=True)
class BackendQubitMapping:
    """Information related to qubit mapping of a backend."""

    #: Mapping of qubit indices (from → to)
    mapping: Mapping[int, int]

    @cached_property
    def circuit_transpiler(self) -> QubitRemappingTranspiler:
        """Returns a :class:`~QubitRemappingTranspiler` that remaps the qubit
        indices in the circuit."""
        return QubitRemappingTranspiler(self.mapping)

    @cached_property
    def _reverse_bit_map(self) -> Mapping[int, int]:
        return _create_reverse_map(self.mapping)

    def unmap_sampling_counts(self, m: SamplingCounts) -> SamplingCounts:
        """Converts :class:`~SamplingCounts` obtained from the backend so that
        the keys (measurement results) are represented in terms of qubits
        before mapping."""
        return _reverse_map_counts(m, self._reverse_bit_map)


@dataclass(frozen=True)
class QubitMappedSamplingResult(SamplingResult):
    """:class:`~SamplingResult` that takes the qubit mapping into consideration.

    Given a "raw" sampling result from the backend, returns a converted result
    with the keys (measurement results) represented in terms of qubits before
    mapping."""

    sampling_result: SamplingResult
    qubit_mapping: BackendQubitMapping

    @property
    def counts(self) -> SamplingCounts:
        return self.qubit_mapping.unmap_sampling_counts(self.sampling_result.counts)


@dataclass(frozen=True)
class QubitMappedSamplingJob(SamplingJob):
    """:class:`~SamplingJob` that takes the qubit mapping into consideration.

    Given a "raw" sampling job from the backend, returns a converted job
    which returns a :class:`~QubitMappedSamplingResult`."""

    sampling_job: SamplingJob
    qubit_mapping: BackendQubitMapping

    def result(self) -> QubitMappedSamplingResult:
        return QubitMappedSamplingResult(self.sampling_job.result(), self.qubit_mapping)
