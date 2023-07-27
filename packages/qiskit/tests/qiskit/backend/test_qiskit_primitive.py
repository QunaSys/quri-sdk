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
import random
import re
import string
from collections import Counter
from typing import Any
from unittest.mock import MagicMock

import pytest
import qiskit
from pydantic.json import pydantic_encoder
from qiskit.primitives import SamplerResult
from qiskit.result import QuasiDistribution
from qiskit_ibm_runtime import Options, QiskitRuntimeService
from qiskit_ibm_runtime.runtime_job import JobStatus, RuntimeJob

from quri_parts.backend import BackendError, CompositeSamplingJob
from quri_parts.circuit import QuantumCircuit
from quri_parts.qiskit.backend import (
    QiskitRuntimeSamplingBackend,
    QiskitRuntimeSavedDataSamplingResult,
    QiskitSavedDataSamplingJob,
    Tracker,
)
from quri_parts.qiskit.backend.primitive import (
    QiskitRuntimeSamplingJob,
    QiskitRuntimeSamplingResult,
)
from quri_parts.qiskit.circuit import convert_circuit

from .mock.ibm_runtime_service_mock import mock_get_backend


def fake_run(**kwargs) -> RuntimeJob:  # type: ignore
    jjob = MagicMock(spec=RuntimeJob)

    def _always_false() -> bool:
        return False

    def _status() -> JobStatus:
        return JobStatus.DONE

    def _job_id() -> str:
        return "aaa"

    def result() -> SamplerResult:
        return SamplerResult(
            quasi_dists=[QuasiDistribution({1: 1.0})], metadata=[{"shots": 10}]
        )

    jjob.status = _status
    jjob.running = _always_false
    jjob.job_id = _job_id
    jjob.result = result

    return jjob


def fake_dynamic_run(**kwargs) -> RuntimeJob:  # type: ignore
    jjob = MagicMock(spec=RuntimeJob)

    jjob._status = JobStatus.RUNNING
    jjob._job_id = "".join([random.choice(string.ascii_lowercase) for _ in range(10)])

    def _status() -> JobStatus:
        return jjob._status

    def _job_id() -> str:
        return str(jjob._job_id)

    def _metrics() -> dict[str, Any]:
        if jjob._status != JobStatus.DONE:
            return {}
        return {"usage": {"seconds": 10}}

    def _set_status(job_response: dict[Any, JobStatus]) -> None:
        jjob._status = list(job_response.values())[0]

    def _cancel() -> None:
        jjob._status = JobStatus.CANCELLED

    jjob.status = _status
    jjob.job_id = _job_id
    jjob.metrics = _metrics
    jjob._set_status = _set_status
    jjob.cancel = _cancel

    return jjob


class TestQiskitPrimitive:
    def test_sampler_call(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()
        sampler = QiskitRuntimeSamplingBackend(backend=backend, service=service)

        circuit = QuantumCircuit(5)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitRuntimeSamplingJob)

        assert job._qiskit_job.job_id() == "aaa"
        assert job._qiskit_job.status() == JobStatus.DONE
        assert not job._qiskit_job.running()

        result = job.result()
        assert isinstance(result, QiskitRuntimeSamplingResult)
        assert len(result._qiskit_result.metadata) == 1

    def test_sampler_session(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()

        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            circuit = QuantumCircuit(5)
            job = sampler.sample(circuit, 10)
            sampler.close()

        assert isinstance(job, QiskitRuntimeSamplingJob)

        assert job._qiskit_job.job_id() == "aaa"
        assert job._qiskit_job.status() == JobStatus.DONE
        assert not job._qiskit_job.running()

        result = job.result()
        assert isinstance(result, QiskitRuntimeSamplingResult)
        assert len(result._qiskit_result.metadata) == 1

        # {1: 1.0} * 10
        assert result.counts == {1: 10.0}

        # Checking if the session is closed
        service._api_client.close_session.assert_called_once_with("aaa")

    def test_sampler_composite(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()

        sampler = QiskitRuntimeSamplingBackend(backend=backend, service=service)

        # Setting the max shots for the sake of testing
        sampler._max_shots = 5

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, CompositeSamplingJob)
        result = job.result()
        counts = result.counts

        # Same job is returned twice, making it 20.
        expected_counts = {1: 20.0}
        assert counts == expected_counts

    @pytest.mark.api
    def test_sampler_live_simple(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        sampler = QiskitRuntimeSamplingBackend(backend=backend, service=service)

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitRuntimeSamplingJob)
        result = job.result()
        counts = result.counts

        expected_counts = {1: 10}
        assert counts == expected_counts

    @pytest.mark.api
    def test_sampler_live_composite(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        sampler = QiskitRuntimeSamplingBackend(backend=backend, service=service)

        # Setting the max shots for the sake of testing
        sampler._max_shots = 5

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        job = sampler.sample(circuit, 10)

        assert isinstance(job, CompositeSamplingJob)
        result = job.result()
        counts = result.counts

        expected_counts = {1: 10}
        assert counts == expected_counts

    @pytest.mark.api
    def test_sampler_live_simple_session(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            job = sampler.sample(circuit, 10)

            assert isinstance(job, QiskitRuntimeSamplingJob)
            result = job.result()
            counts = result.counts

            expected_counts = {1: 10}
            assert counts == expected_counts

    @pytest.mark.api
    def test_sampler_live_composite_session(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)

        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            # Setting the max shots for the sake of testing
            sampler._max_shots = 5
            job = sampler.sample(circuit, 10)

            assert isinstance(job, CompositeSamplingJob)
            result = job.result()
            counts = result.counts

            expected_counts = {1: 10}
            assert counts == expected_counts

    def test_sampler_options_dict(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        backend = service.backend()

        options_dict = {"resilience_level": 1, "optimization_level": 2}

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, sampler_options=options_dict
        )

        saved_options = sampler._qiskit_sampler_options

        assert isinstance(saved_options, Options)
        assert saved_options.resilience_level == 1
        assert saved_options.optimization_level == 2

    def test_sampler_options_qiskit_options(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        backend = service.backend()

        qiskit_options = Options()
        qiskit_options.resilience_level = 1
        qiskit_options.optimization_level = 2

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, sampler_options=qiskit_options
        )

        saved_options = sampler._qiskit_sampler_options

        assert isinstance(saved_options, Options)
        assert saved_options.resilience_level == 1
        assert saved_options.optimization_level == 2

    def test_sampler_options_none(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        backend = service.backend()

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, sampler_options=None
        )

        assert sampler._qiskit_sampler_options is None

    def test_saving_mode(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, save_data_while_sampling=True
        )
        sampler._max_shots = 10

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
            convert_circuit(qp_circuit).measure_all(False), backend
        )
        circuit_qasm = qiskit_transpiled_circuit.qasm()

        sampling_job = sampler.sample(qp_circuit, 20)
        measurement_counter = sampling_job.result().counts

        # Check if the circuit string and n_shots are distribited correctly
        measurement_counter_from_memory: Counter[int] = Counter()
        for qasm_str, n_shots, saved_job in sampler._saved_data:
            measurement_counter_from_memory += Counter(saved_job.result().counts)

            assert qasm_str == circuit_qasm
            assert n_shots == 10

        # Check if the saved data counter sums up to the correct counter.
        assert measurement_counter_from_memory == measurement_counter

        # Construct the saved data objects
        quasi_dist_0 = sampler._saved_data[0][2]._qiskit_job.result().quasi_dists[0]
        quasi_dist_1 = sampler._saved_data[1][2]._qiskit_job.result().quasi_dists[0]

        expected_saved_data_0 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitRuntimeSavedDataSamplingResult(
                quasi_dist=quasi_dist_0, n_shots=10
            ),
        )

        expected_saved_data_1 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitRuntimeSavedDataSamplingResult(
                quasi_dist=quasi_dist_1, n_shots=10
            ),
        )

        # Check the jobs and jobs_json properties.
        expected_saved_data_seq = [
            expected_saved_data_0,
            expected_saved_data_1,
        ]
        assert sampler.jobs == expected_saved_data_seq
        expected_json_str = json.dumps(
            expected_saved_data_seq, default=pydantic_encoder
        )
        assert sampler.jobs_json == expected_json_str

    def test_saving_mode_session(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_run
        backend = service.backend()

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
            convert_circuit(qp_circuit).measure_all(False), backend
        )
        circuit_qasm = qiskit_transpiled_circuit.qasm()

        with QiskitRuntimeSamplingBackend(
            backend=backend, service=service, save_data_while_sampling=True
        ) as sampler:
            sampling_job = sampler.sample(qp_circuit, 10)
            sampler.close()

        measurement_counter = sampling_job.result().counts

        # Check if the circuit string and n_shots are distribited correctly
        measurement_counter_from_memory: Counter[int] = Counter()
        for qasm_str, n_shots, saved_job in sampler._saved_data:
            measurement_counter_from_memory += Counter(saved_job.result().counts)

            assert qasm_str == circuit_qasm
            assert n_shots == 10

        # Check if the saved data counter sums up to the correct counter.
        assert measurement_counter_from_memory == measurement_counter

        # Construct the saved data objects
        quasi_dist = sampler._saved_data[0][2]._qiskit_job.result().quasi_dists[0]

        expected_saved_data = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitRuntimeSavedDataSamplingResult(
                quasi_dist=quasi_dist, n_shots=10
            ),
        )

        # Check the jobs and jobs_json properties.
        expected_saved_data_seq = [expected_saved_data]
        assert sampler.jobs == expected_saved_data_seq
        expected_json_str = json.dumps(
            expected_saved_data_seq, default=pydantic_encoder
        )
        assert sampler.jobs_json == expected_json_str

    @pytest.mark.api
    def test_live_saving_mode(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, save_data_while_sampling=True
        )
        sampler._max_shots = 10

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
            convert_circuit(qp_circuit).measure_all(False), backend
        )
        circuit_qasm = qiskit_transpiled_circuit.qasm()

        sampling_job = sampler.sample(qp_circuit, 20)
        measurement_counter = sampling_job.result().counts

        # Check if the circuit string and n_shots are distribited correctly
        measurement_counter_from_memory: Counter[int] = Counter()
        for qasm_str, n_shots, saved_job in sampler._saved_data:
            measurement_counter_from_memory += Counter(saved_job.result().counts)

            assert qasm_str == circuit_qasm
            assert n_shots == 10

        # Check if the saved data counter sums up to the correct counter.
        assert measurement_counter_from_memory == measurement_counter

        # Construct the saved data objects
        quasi_dist_0 = sampler._saved_data[0][2]._qiskit_job.result().quasi_dists[0]
        quasi_dist_1 = sampler._saved_data[1][2]._qiskit_job.result().quasi_dists[0]

        expected_saved_data_0 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitRuntimeSavedDataSamplingResult(
                quasi_dist=quasi_dist_0, n_shots=10
            ),
        )

        expected_saved_data_1 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitRuntimeSavedDataSamplingResult(
                quasi_dist=quasi_dist_1, n_shots=10
            ),
        )

        # Check the jobs and jobs_json properties.
        expected_saved_data_seq = [
            expected_saved_data_0,
            expected_saved_data_1,
        ]
        assert sampler.jobs == expected_saved_data_seq
        expected_json_str = json.dumps(
            expected_saved_data_seq, default=pydantic_encoder
        )
        assert sampler.jobs_json == expected_json_str

    @pytest.mark.api
    def test_live_saving_mode_session(self) -> None:
        service = QiskitRuntimeService()
        backend = service.backend("ibmq_qasm_simulator")

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
            convert_circuit(qp_circuit).measure_all(False), backend
        )
        circuit_qasm = qiskit_transpiled_circuit.qasm()

        with QiskitRuntimeSamplingBackend(
            backend=backend, service=service, save_data_while_sampling=True
        ) as sampler:
            sampling_job = sampler.sample(qp_circuit, 10)
            sampler.close()

        measurement_counter = sampling_job.result().counts

        # Check if the circuit string and n_shots are distribited correctly
        measurement_counter_from_memory: Counter[int] = Counter()
        for qasm_str, n_shots, saved_job in sampler._saved_data:
            measurement_counter_from_memory += Counter(saved_job.result().counts)

            assert qasm_str == circuit_qasm
            assert n_shots == 10

        # Check if the saved data counter sums up to the correct counter.
        assert measurement_counter_from_memory == measurement_counter

        # Construct the saved data objects
        quasi_dist = sampler._saved_data[0][2]._qiskit_job.result().quasi_dists[0]

        expected_saved_data = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitRuntimeSavedDataSamplingResult(
                quasi_dist=quasi_dist, n_shots=10
            ),
        )

        # Check the jobs and jobs_json properties.
        expected_saved_data_seq = [expected_saved_data]
        assert sampler.jobs == expected_saved_data_seq
        expected_json_str = json.dumps(
            expected_saved_data_seq, default=pydantic_encoder
        )
        assert sampler.jobs_json == expected_json_str

    def test_reject_job(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_dynamic_run
        backend = service.backend()
        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, total_time_limit=5
        )
        job1 = sampling_backend.sample(QuantumCircuit(2), 100)
        assert isinstance(job1, QiskitRuntimeSamplingJob)

        job2 = sampling_backend.sample(QuantumCircuit(2), 100)
        assert isinstance(job2, QiskitRuntimeSamplingJob)

        job1._qiskit_job._set_status({"new_job_status": JobStatus.DONE})

        with pytest.raises(
            BackendError,
            match=re.escape(
                "Qiskit Device run failed. Failed reason:\n"
                "The submission of this job is aborted due to "
                "run time limit of 5 seconds is exceeded. "
                "Other unfinished jobs are also aborted."
            ),
        ):
            sampling_backend.sample(QuantumCircuit(2), 100)

        assert job2._qiskit_job.status() == JobStatus.CANCELLED

    def test_job_registered_to_tracker(self) -> None:
        runtime_service = mock_get_backend("FakeVigo")
        service = runtime_service()
        service.run = fake_dynamic_run
        backend = service.backend()
        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, total_time_limit=5000
        )
        assert isinstance(sampling_backend.tracker, Tracker)

        job1 = sampling_backend.sample(QuantumCircuit(2), 100)
        assert sampling_backend.tracker.running_jobs == [job1]

        job2 = sampling_backend.sample(QuantumCircuit(2), 100)
        assert sampling_backend.tracker.running_jobs == [job1, job2]
