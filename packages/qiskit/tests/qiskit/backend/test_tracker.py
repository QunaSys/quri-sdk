# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any
from unittest.mock import MagicMock

import pytest
from qiskit_ibm_runtime.runtime_job import JobStatus, RuntimeJob

from quri_parts.qiskit.backend.primitive import QiskitRuntimeSamplingJob
from quri_parts.qiskit.backend.tracker import Tracker

from .mock.ibm_runtime_service_mock import mock_get_backend


def fake_dynamic_run(job_id: str, seconds: float, **kwargs: Any) -> RuntimeJob:
    jjob = MagicMock(spec=RuntimeJob)

    jjob._status = "RUNNING"
    jjob._job_id = job_id

    def _status() -> JobStatus:
        return jjob._status

    def _job_id() -> str:
        return str(jjob._job_id)

    def _metrics() -> dict[str, Any]:
        if jjob._status != "DONE":
            return {}
        return {"usage": {"seconds": seconds}}

    def _set_status(job_response: dict[Any, str]) -> None:
        jjob._status = list(job_response.values())[0]

    jjob.status = _status
    jjob.job_id = _job_id
    jjob.metrics = _metrics
    jjob._set_status = _set_status

    return jjob


def test_add_job_for_tracking() -> None:
    runtime_service = mock_get_backend()
    service = runtime_service()

    tracker = Tracker()
    assert tracker.finished_jobs == []
    assert tracker.running_jobs == []

    # Register job 1
    service.run = partial(fake_dynamic_run, "aaa", 10)
    job1 = QiskitRuntimeSamplingJob(qiskit_job=service.run())
    tracker.add_job_for_tracking(job1)
    assert tracker.finished_jobs == []
    assert tracker.running_jobs == [job1]

    # Register job 2
    service.run = partial(fake_dynamic_run, "bbb", 10)
    job2 = QiskitRuntimeSamplingJob(qiskit_job=service.run())
    tracker.add_job_for_tracking(job2)
    assert tracker.finished_jobs == []
    assert tracker.running_jobs == [job1, job2]

    # Register bad job
    with pytest.raises(AssertionError, match="This job is already submitted before."):
        bad_job_id_job = QiskitRuntimeSamplingJob(qiskit_job=service.run())
        tracker.add_job_for_tracking(bad_job_id_job)

    assert tracker.running_jobs == [job1, job2]

    # Register new job after all jobs are done without exceeding time limit.
    tracker._finished_jobs = {"aaa": job1, "bbb": job2}
    tracker._running_jobs = {}

    service.run = partial(fake_dynamic_run, "ccc", 10)
    job3 = QiskitRuntimeSamplingJob(qiskit_job=service.run())
    tracker.add_job_for_tracking(job3)

    assert tracker.finished_jobs == [job1, job2]
    assert tracker.running_jobs == [job3]


def test_total_run_time() -> None:
    runtime_service = mock_get_backend()
    service = runtime_service()

    tracker = Tracker()

    service.run = partial(fake_dynamic_run, "aaa", 10)
    job1 = QiskitRuntimeSamplingJob(qiskit_job=service.run())

    service.run = partial(fake_dynamic_run, "bbb", 50)
    job2 = QiskitRuntimeSamplingJob(qiskit_job=service.run())

    service.run = partial(fake_dynamic_run, "ccc", 80)
    job3 = QiskitRuntimeSamplingJob(qiskit_job=service.run())

    service.run = partial(fake_dynamic_run, "ddd", 20)
    job4 = QiskitRuntimeSamplingJob(qiskit_job=service.run())

    tracker._running_jobs["aaa"] = job1
    tracker._running_jobs["bbb"] = job2
    tracker._running_jobs["ccc"] = job3
    tracker._running_jobs["ddd"] = job4

    assert tracker.total_run_time == 0

    # Backend finishes Job 1 execution
    job1._qiskit_job._set_status({"new job status": "DONE"})
    assert tracker.total_run_time == 10.0
    assert tracker.running_jobs == [job2, job3, job4]
    assert tracker.finished_jobs == [job1]

    # Backend finishes Job 2 execution
    job2._qiskit_job._set_status({"new job status": "DONE"})

    assert tracker.total_run_time == 60.0
    assert tracker.running_jobs == [job3, job4]
    assert tracker.finished_jobs == [job1, job2]

    # Backend finishes Job 3 execution
    job3._qiskit_job._set_status({"new job status": "DONE"})

    assert tracker.total_run_time == 140.0
    assert tracker.running_jobs == [job4]
    assert tracker.finished_jobs == [job1, job2, job3]
