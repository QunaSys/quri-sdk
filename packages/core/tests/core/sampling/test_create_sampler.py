# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from unittest import mock

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.sampling import (
    MeasurementCounts,
    create_concurrent_sampler_from_sampling_backend,
    create_sampler_from_sampling_backend,
)


def create_mock_backend(counts: Sequence[MeasurementCounts]) -> mock.Mock:
    def create_job(c: MeasurementCounts) -> mock.Mock:
        job = mock.Mock()
        result = job.result.return_value
        result.counts = c
        return job

    backend = mock.Mock()
    backend.sample.side_effect = [create_job(c) for c in counts]

    return backend


def test_create_sampler_from_sampling_backend() -> None:
    counts = {0: 100, 1: 200, 2: 300, 3: 400}
    backend = create_mock_backend([counts])

    sampler = create_sampler_from_sampling_backend(backend)
    circuit = QuantumCircuit(3)
    sampling_result = sampler(circuit, 1000)

    assert sampling_result == counts
    backend.sample.assert_called_with(circuit, 1000)


def test_create_concurrent_sampler_from_sampling_backend() -> None:
    counts = [
        {0: 100, 1: 200, 2: 300, 3: 400},
        {0: 300, 1: 300, 2: 200, 3: 100},
    ]
    backend = create_mock_backend(counts)

    sampler = create_concurrent_sampler_from_sampling_backend(backend)
    circuits = [QuantumCircuit(3), QuantumCircuit(2)]
    sampling_results = sampler([(circuits[0], 1000), (circuits[1], 900)])

    assert list(sampling_results) == counts
    assert backend.sample.call_args_list == [
        mock.call(circuits[0], 1000),
        mock.call(circuits[1], 900),
    ]
