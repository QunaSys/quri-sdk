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
import warnings
from collections import Counter
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import qiskit
from pydantic.json import pydantic_encoder
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import qasm3
from qiskit.primitives import PrimitiveResult, PubResult
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJob, SamplerOptions
from qiskit_ibm_runtime.runtime_job import JobStatus

from quri_parts.backend import BackendError, CompositeSamplingJob
from quri_parts.circuit import QuantumCircuit
from quri_parts.qiskit.backend import (
    QiskitRuntimeSamplingBackend,
    QiskitSavedDataSamplingJob,
    QiskitSavedDataSamplingResult,
    Tracker,
)
from quri_parts.qiskit.backend.primitive import (
    QiskitRuntimeSamplingJob,
    QiskitRuntimeSamplingResult,
)
from quri_parts.qiskit.circuit import convert_circuit

from .mock.ibm_runtime_service_mock import mock_get_backend


def fake_validate(*args, **kwargs) -> None:  # type: ignore
    return


def fake_run(*args, **kwargs) -> RuntimeJob:  # type: ignore
    jjob = MagicMock(spec=RuntimeJob)

    def _always_false() -> bool:
        return False

    def _status() -> JobStatus:
        return JobStatus.DONE

    def _job_id() -> str:
        return "aaa"

    from unittest.mock import Mock

    def result() -> PrimitiveResult:
        pubres = Mock(spec=PubResult)
        pubres.data.meas.get_counts.return_value = {"000001": 10}
        return PrimitiveResult([pubres])

    jjob.status = _status
    jjob.running = _always_false
    jjob.job_id = _job_id
    jjob.result = result
    return jjob


def fake_dynamic_run(*args, **kwargs) -> RuntimeJob:  # type: ignore
    jjob = MagicMock(spec=RuntimeJob)

    jjob._status = "RUNNING"
    jjob._job_id = "".join([random.choice(string.ascii_lowercase) for _ in range(10)])

    def _status() -> JobStatus:
        return jjob._status

    def _job_id() -> str:
        return str(jjob._job_id)

    def _metrics() -> dict[str, Any]:
        if jjob._status != "DONE":
            return {}
        return {"usage": {"seconds": 10}}

    def _set_status(job_response: dict[Any, str]) -> None:
        jjob._status = list(job_response.values())[0]

    def _cancel() -> None:
        jjob._status = "CANCELLED"

    jjob.status = _status
    jjob.job_id = _job_id
    jjob.metrics = _metrics
    jjob._set_status = _set_status
    jjob.cancel = _cancel

    return jjob


def fake_qiskit_transpile(*args: Any, **kwargs: Any) -> QiskitQuantumCircuit:
    circuit = QiskitQuantumCircuit(2)
    circuit.measure_all(inplace=True)
    return circuit


class TestQiskitPrimitive:
    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_sampler_call(self) -> None:
        runtime_service = mock_get_backend()
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

    @patch("qiskit_ibm_runtime.Session._create_session", return_value="aaa")
    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_sampler_session(self, _: str) -> None:
        runtime_service = mock_get_backend(False)
        service = runtime_service
        service.run = fake_run
        backend = service.backend()

        with QiskitRuntimeSamplingBackend(backend=backend, service=service) as sampler:
            circuit = QuantumCircuit(5)
            job = sampler.sample(circuit, 10)

        assert isinstance(job, QiskitRuntimeSamplingJob)

        assert job._qiskit_job.job_id() == "aaa"
        assert job._qiskit_job.status() == JobStatus.DONE
        assert not job._qiskit_job.running()

        result = job.result()
        assert isinstance(result, QiskitRuntimeSamplingResult)

        # {1: 1.0} * 10
        assert result.counts == {1: 10.0}

        # Checking if the session is closed
        service._api_client.close_session.assert_called()

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_sampler_composite(self) -> None:
        runtime_service = mock_get_backend()
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
        assert isinstance(job._qiskit_job, RuntimeJob)
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

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_sampler_options_dict(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        options_dict = {"default_shots": 2}

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, sampler_options=options_dict
        )

        saved_options = sampler._qiskit_sampler_options

        assert isinstance(saved_options, SamplerOptions)
        assert saved_options.default_shots == 2

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_sampler_options_qiskit_options(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        qiskit_options = SamplerOptions()
        qiskit_options.environment.job_tags = ["tag"]

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, sampler_options=qiskit_options
        )

        saved_options = sampler._qiskit_sampler_options

        assert isinstance(saved_options, SamplerOptions)
        assert saved_options.environment.job_tags == ["tag"]

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_sampler_options_none(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        sampler = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, sampler_options=None
        )

        assert sampler._qiskit_sampler_options is None

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_saving_mode(self) -> None:
        runtime_service = mock_get_backend()
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
        circuit_qasm = qasm3.dumps(qiskit_transpiled_circuit)

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
        qiskit_result_0 = sampler._saved_data[0][2]._qiskit_job.result()
        qiskit_sampling_cnt_0 = qiskit_result_0[0].data.meas.get_counts()
        qiskit_result_1 = sampler._saved_data[1][2]._qiskit_job.result()
        qiskit_sampling_cnt_1 = qiskit_result_1[0].data.meas.get_counts()

        expected_saved_data_0 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitSavedDataSamplingResult(qiskit_sampling_cnt_0),
        )

        expected_saved_data_1 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitSavedDataSamplingResult(qiskit_sampling_cnt_1),
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

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_saving_mode_session(self) -> None:
        runtime_service = mock_get_backend()
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
        circuit_qasm = qasm3.dumps(qiskit_transpiled_circuit)

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
        qiskit_result = sampler._saved_data[0][2]._qiskit_job.result()
        sampling_cnt = qiskit_result[0].data.meas.get_counts()

        expected_saved_data = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitSavedDataSamplingResult(sampling_cnt),
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
        circuit_qasm = qasm3.dumps(qiskit_transpiled_circuit)

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
        result_0 = sampler._saved_data[0][2]._qiskit_job.result()
        cnt_0 = result_0[0].data.meas.get_counts()
        result_1 = sampler._saved_data[1][2]._qiskit_job.result()
        cnt_1 = result_1[0].data.meas.get_counts()

        expected_saved_data_0 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitSavedDataSamplingResult(cnt_0),
        )

        expected_saved_data_1 = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitSavedDataSamplingResult(cnt_1),
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
        circuit_qasm = qasm3.dumps(qiskit_transpiled_circuit)

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
        result = sampler._saved_data[0][2]._qiskit_job.result()
        cnt = result[0].data.meas.get_counts()

        expected_saved_data = QiskitSavedDataSamplingJob(
            circuit_qasm=circuit_qasm,
            n_shots=10,
            saved_result=QiskitSavedDataSamplingResult(cnt),
        )

        # Check the jobs and jobs_json properties.
        expected_saved_data_seq = [expected_saved_data]
        assert sampler.jobs == expected_saved_data_seq
        expected_json_str = json.dumps(
            expected_saved_data_seq, default=pydantic_encoder
        )
        assert sampler.jobs_json == expected_json_str

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_reject_job(self) -> None:
        runtime_service = mock_get_backend()
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

        job1._qiskit_job._set_status({"new_job_status": "DONE"})

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

        assert job2._qiskit_job.status() == "CANCELLED"

    @patch("qiskit.transpile", fake_qiskit_transpile)
    def test_job_registered_to_tracker(self) -> None:
        runtime_service = mock_get_backend()
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

    def test_get_batch_execution_time_not_divisible(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend,
            service=service,
            single_job_max_execution_time=500,
            total_time_limit=1000,
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50]
        ) == (500 // 3, 1000 // 3)
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 1]
        ) == (500 // 3, 1000 // 3)
        assert sampling_backend._get_batch_execution_time_and_time_left([50]) == (
            500,
            1000,
        )

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, total_time_limit=1000
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50]
        ) == (None, 1000 // 3)
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 1]
        ) == (None, 1000 // 3)
        assert sampling_backend._get_batch_execution_time_and_time_left([50]) == (
            None,
            1000,
        )

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, single_job_max_execution_time=500
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50]
        ) == (500 // 3, None)
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 1]
        ) == (500 // 3, None)
        assert sampling_backend._get_batch_execution_time_and_time_left([50]) == (
            500,
            None,
        )

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 50]
        ) == (None, None)

    def test_get_batch_execution_time(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend,
            service=service,
            single_job_max_execution_time=500,
            total_time_limit=1000,
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 50]
        ) == (500 / 4, 1000 / 4)
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 1]
        ) == (500 / 4, 1000 / 4)
        assert sampling_backend._get_batch_execution_time_and_time_left([50]) == (
            500,
            1000,
        )

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, total_time_limit=1000
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 50]
        ) == (None, 1000 / 4)
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 1]
        ) == (None, 1000 / 4)
        assert sampling_backend._get_batch_execution_time_and_time_left([50]) == (
            None,
            1000,
        )

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, single_job_max_execution_time=500
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 50]
        ) == (500 / 4, None)
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 1]
        ) == (500 / 4, None)
        assert sampling_backend._get_batch_execution_time_and_time_left([50]) == (
            500,
            None,
        )

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service
        )
        assert sampling_backend._get_batch_execution_time_and_time_left(
            [50, 50, 50, 50]
        ) == (None, None)

    def test_get_sampler_option_with_max_execution_time_limit(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, single_job_max_execution_time=500
        )
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=30, batch_time_left=None
            ).max_execution_time
            == 300
        )
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=400, batch_time_left=None
            ).max_execution_time
            == 400
        )

    def test_get_sampler_option_with_tracker_time_limit(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend,
            service=service,
        )
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=None, batch_time_left=3000
            ).max_execution_time
            == 3000
        )

        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=None, batch_time_left=30
            ).max_execution_time
            == 300
        )

    def test_get_sampler_option_with_both_time_limit(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()

        # Test sampling backend with both total_time_limit and
        # single_job_max_execution_time settings.
        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend,
            service=service,
        )

        # tl > tb > 300
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=700, batch_time_left=800
            ).max_execution_time
            == 700
        )

        # tb > tl > 300
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=700, batch_time_left=600
            ).max_execution_time
            == 600
        )

        # tl  > 300 > tb
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=70, batch_time_left=600
            ).max_execution_time
            == 300
        )

        # tb > 300 > tl
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=350, batch_time_left=100
            ).max_execution_time
            == 300
        )

        # 300 > tb > tl
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=140, batch_time_left=100
            ).max_execution_time
            == 300
        )

        # 300 > tl > tb
        assert (
            sampling_backend._get_sampler_option_with_time_limit(
                batch_exe_time=10, batch_time_left=100
            ).max_execution_time
            == 300
        )

    def test_check_execution_time_limitability_strict(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()
        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, strict_time_limit=True
        )

        with pytest.raises(
            BackendError,
            match="Max execution time limit of 30 seconds cannot be followed strictly.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=30, batch_time_left=None
            )

        with pytest.raises(
            BackendError,
            match="Max execution time limit of 30 seconds cannot be followed strictly.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=None, batch_time_left=30
            )

        with pytest.raises(
            BackendError,
            match="Max execution time limit of 50 seconds cannot be followed strictly.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=30, batch_time_left=50
            )

        with pytest.raises(
            BackendError,
            match="Max execution time limit of 50 seconds cannot be followed strictly.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=80, batch_time_left=50
            )

        # test no excpetion raised
        try:
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=300, batch_time_left=300
            )
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=800, batch_time_left=500
            )
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=800, batch_time_left=None
            )
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=None, batch_time_left=500
            )
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=None, batch_time_left=None
            )
        except:  # noqa: E722
            assert False

    def test_check_execution_time_limitability_non_strict(self) -> None:
        runtime_service = mock_get_backend()
        service = runtime_service()
        backend = service.backend()
        sampling_backend = QiskitRuntimeSamplingBackend(
            backend=backend, service=service, strict_time_limit=False
        )

        with pytest.warns(
            UserWarning,
            match="The time limit of 30 seconds is likely going to be exceeded.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=30, batch_time_left=None
            )

        with pytest.warns(
            UserWarning,
            match="The time limit of 30 seconds is likely going to be exceeded.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=None, batch_time_left=30
            )

        with pytest.warns(
            UserWarning,
            match="The time limit of 50 seconds is likely going to be exceeded.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=30, batch_time_left=50
            )

        with pytest.warns(
            UserWarning,
            match="The time limit of 80 seconds is likely going to be exceeded.",
        ):
            sampling_backend._check_execution_time_limitability(
                batch_exe_time=80, batch_time_left=50
            )

        # test no warnings raised
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sampling_backend._check_execution_time_limitability(300, 300)
            sampling_backend._check_execution_time_limitability(800, 500)
            sampling_backend._check_execution_time_limitability(800, None)
            sampling_backend._check_execution_time_limitability(None, 500)
            sampling_backend._check_execution_time_limitability(None, None)
