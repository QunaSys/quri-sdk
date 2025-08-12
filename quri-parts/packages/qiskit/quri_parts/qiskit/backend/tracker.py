# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Sequence

# from qiskit_ibm_runtime.runtime_job_v2 import JobStatus

if TYPE_CHECKING:
    from .primitive import QiskitRuntimeSamplingJob


class Tracker:
    """A tracker that tracks how much billable time is used by jobs executed on
    IBM Backends."""

    def __init__(self) -> None:
        self._total_time = 0.0

        self._running_jobs: dict[
            str, QiskitRuntimeSamplingJob
        ] = {}  # The key is job id string
        self._finished_jobs: dict[
            str, QiskitRuntimeSamplingJob
        ] = {}  # The key is job id string

    @property
    def total_run_time(self) -> float:
        """The total run time that is billable."""
        self._track()
        return self._total_time

    @property
    def finished_jobs(self) -> Sequence["QiskitRuntimeSamplingJob"]:
        """All the jobs that are finished by the IBM backend."""
        self._track()
        return list(self._finished_jobs.values())

    @property
    def running_jobs(self) -> Sequence["QiskitRuntimeSamplingJob"]:
        """All the jobs that are still runnign on the IBM backend."""
        self._track()
        return list(self._running_jobs.values())

    def _track(self) -> None:
        finished_id = []
        # Scan through running jobs to see if there are finished ones.
        for job in self._running_jobs.values():
            job_id = job._qiskit_job.job_id()
            metrics = job._qiskit_job.metrics()
            finished = job._qiskit_job.status() == "DONE"
            if finished:
                finished_id.append(job_id)
                self._finished_jobs[job_id] = job
                self._total_time += metrics["usage"]["seconds"]

        # Remove finished jobs from running_jobs
        for job_id in finished_id:
            del self._running_jobs[job_id]

    def add_job_for_tracking(self, runtime_job: "QiskitRuntimeSamplingJob") -> None:
        """Register :class:`~QiskitRuntimeSamplingJob` to the tracker for
        tracking billable job execution time."""
        assert (
            runtime_job._qiskit_job.job_id() not in self._running_jobs
            and runtime_job._qiskit_job.job_id() not in self._finished_jobs
        ), "This job is already submitted before."

        self._running_jobs[runtime_job._qiskit_job.job_id()] = runtime_job
