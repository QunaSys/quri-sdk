# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest
from qiskit.primitives import SamplerResult
from qiskit.result import QuasiDistribution
from qiskit_ibm_runtime import Options, QiskitRuntimeService
from qiskit_ibm_runtime.runtime_job import JobStatus, RuntimeJob

from quri_parts.backend import CompositeSamplingJob
from quri_parts.circuit import QuantumCircuit
from quri_parts.qiskit.backend import QiskitRuntimeSamplingBackend
from quri_parts.qiskit.backend.primitive import (
    QiskitRuntimeSamplingJob,
    QiskitRuntimeSamplingResult,
)

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
