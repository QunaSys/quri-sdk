# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from quri_parts.backend import CompositeSamplingJob, SamplingCounts


def create_mock_job(counts: SamplingCounts) -> mock.Mock:
    job = mock.Mock()
    result = job.result.return_value
    result.counts = counts
    return job


def test_composite_sampling_job() -> None:
    counts = [
        {0: 100, 1: 200, 2: 300, 3: 400},
        {0: 200, 1: 400, 3: 100, 4: 500},
    ]
    jobs = [create_mock_job(c) for c in counts]
    composite_job = CompositeSamplingJob(jobs=jobs)
    assert composite_job.result().counts == {0: 300, 1: 600, 2: 300, 3: 500, 4: 500}
