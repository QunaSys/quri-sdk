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
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

qnx = pytest.importorskip("qnexus")

from pytket.backends.backendresult import BackendResult  # noqa: E402
from pytket.circuit import Bit, Circuit  # noqa: E402
from pytket.utils.outcomearray import OutcomeArray  # noqa: E402
from qnexus.models.annotations import Annotations  # noqa: E402
from qnexus.models.job_status import JobStatusEnum  # noqa: E402
from qnexus.models.references import ExecuteJobRef, ProjectRef  # noqa: E402

from quri_parts.backend import BackendError  # noqa: E402
from quri_parts.circuit import QuantumCircuit  # noqa: E402
from quri_parts.quantinuum.backend import (  # noqa: E402
    QuantinuumSamplingBackend,
    QuantinuumSamplingJob,
    QuantinuumSamplingResult,
    backend_result_to_sampling_counts,
    default_backend_config,
    distribute_shots,
)


def _backend_result(readouts: Sequence[Sequence[int]], bits: Sequence[Bit]) -> Any:
    shots = np.array(readouts, dtype=np.uint8)
    return BackendResult(shots=OutcomeArray.from_readouts(shots), c_bits=list(bits))


def _project_ref() -> ProjectRef:
    return ProjectRef(
        id=uuid4(),
        annotations=Annotations(name="test-project"),
        contents_modified=datetime.now(timezone.utc),
    )


def _execute_job_ref(project: ProjectRef) -> ExecuteJobRef:
    return ExecuteJobRef(
        id=uuid4(),
        annotations=Annotations(name="test-execute"),
        last_status=JobStatusEnum.SUBMITTED,
        last_message="",
        project=project,
    )


class FakeNexus:
    """Records calls to the qnexus functions used by the backend."""

    def __init__(self, readouts_per_item: Sequence[Sequence[Sequence[int]]]):
        self.project = _project_ref()
        self.readouts_per_item = readouts_per_item
        self.uploaded: list[Circuit] = []
        self.compile_calls: list[dict[str, Any]] = []
        self.execute_calls: list[dict[str, Any]] = []
        self.waited: list[Any] = []
        self.cancelled: list[Any] = []
        self.compiled_ref = SimpleNamespace(id=uuid4(), name="compiled")
        self.execute_ref = _execute_job_ref(self.project)

    def get_or_create(self, name: str, **kwargs: Any) -> ProjectRef:
        self.project_name = name
        return self.project

    def upload(self, circuit: Circuit, **kwargs: Any) -> Any:
        self.uploaded.append(circuit)
        return SimpleNamespace(id=uuid4(), name=kwargs.get("name"), circuit=circuit)

    def start_compile_job(self, **kwargs: Any) -> Any:
        self.compile_calls.append(kwargs)
        return SimpleNamespace(id=uuid4(), kind="compile")

    def start_execute_job(self, **kwargs: Any) -> ExecuteJobRef:
        self.execute_calls.append(kwargs)
        return self.execute_ref

    def wait_for(self, job: Any, **kwargs: Any) -> Any:
        self.waited.append((job, kwargs))
        return SimpleNamespace(status=JobStatusEnum.COMPLETED)

    def results(self, job: Any, **kwargs: Any) -> list[Any]:
        if getattr(job, "kind", None) == "compile":
            return [SimpleNamespace(get_output=lambda: self.compiled_ref)]
        bits = self.uploaded[-1].bits
        return [
            SimpleNamespace(download_result=lambda r=r: _backend_result(r, bits))
            for r in self.readouts_per_item
        ]

    def cancel(self, job: Any, **kwargs: Any) -> None:
        self.cancelled.append(job)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(qnx.projects, "get_or_create", self.get_or_create)
        monkeypatch.setattr(qnx.circuits, "upload", self.upload)
        monkeypatch.setattr(qnx, "start_compile_job", self.start_compile_job)
        monkeypatch.setattr(qnx, "start_execute_job", self.start_execute_job)
        monkeypatch.setattr(qnx.jobs, "wait_for", self.wait_for)
        monkeypatch.setattr(qnx.jobs, "results", self.results)
        monkeypatch.setattr(qnx.jobs, "cancel", self.cancel)


def _bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    return circuit


class TestHelpers:
    def test_backend_result_to_sampling_counts(self) -> None:
        bits = [Bit("c", 0), Bit("c", 1), Bit("c", 2)]
        result = _backend_result([[1, 0, 0], [1, 0, 0], [0, 1, 1]], bits)
        assert backend_result_to_sampling_counts(result) == {0b001: 2, 0b110: 1}

    def test_backend_result_to_sampling_counts_sorts_bits(self) -> None:
        # bits registered out of order: index 2 first
        bits = [Bit("c", 2), Bit("c", 0), Bit("c", 1)]
        result = _backend_result([[1, 0, 0]], bits)
        assert backend_result_to_sampling_counts(result) == {0b100: 1}

    def test_distribute_shots(self) -> None:
        assert distribute_shots(10, None) == [10]
        assert distribute_shots(10, 10) == [10]
        assert distribute_shots(10, 4) == [4, 4, 2]
        assert distribute_shots(8, 4) == [4, 4]
        with pytest.raises(ValueError):
            distribute_shots(0, None)
        with pytest.raises(ValueError):
            distribute_shots(5, 0)

    def test_default_backend_config(self) -> None:
        config = default_backend_config("H2-1E")
        assert isinstance(config, qnx.QuantinuumConfig)
        assert config.device_name == "H2-1E"

        helios = default_backend_config("Helios-1E")
        assert isinstance(helios, qnx.HeliosConfig)
        assert helios.system_name == "Helios-1E"


class TestQuantinuumSamplingBackend:
    def test_init_requires_device_or_config(self) -> None:
        with pytest.raises(ValueError):
            QuantinuumSamplingBackend()

        backend = QuantinuumSamplingBackend(backend_config=qnx.AerConfig())
        assert isinstance(backend.backend_config, qnx.AerConfig)

    def test_sample_with_compile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0], [1, 1], [1, 1]]])
        fake.install(monkeypatch)

        backend = QuantinuumSamplingBackend(
            "H2-1E",
            project="my-project",
            optimisation_level=1,
            compile_timeout=30.0,
            execute_kwargs={"max_cost": 5.0},
        )
        job = backend.sample(_bell_circuit(), 3)

        assert isinstance(job, QuantinuumSamplingJob)
        assert fake.project_name == "my-project"
        assert backend.project is fake.project

        # uploaded circuit has all qubits measured
        assert len(fake.uploaded) == 1
        uploaded = fake.uploaded[0]
        assert uploaded.n_qubits == 2
        assert len(uploaded.bits) == 2

        assert len(fake.compile_calls) == 1
        compile_call = fake.compile_calls[0]
        assert compile_call["optimisation_level"] == 1
        assert compile_call["backend_config"] is backend.backend_config
        assert compile_call["project"] is fake.project
        assert fake.waited[0][1] == {"timeout": 30.0}

        assert len(fake.execute_calls) == 1
        execute_call = fake.execute_calls[0]
        assert execute_call["programs"] == [fake.compiled_ref]
        assert execute_call["n_shots"] == [3]
        assert execute_call["backend_config"] is backend.backend_config
        assert execute_call["max_cost"] == 5.0

        result = job.result()
        assert isinstance(result, QuantinuumSamplingResult)
        assert result.counts == {0b00: 1, 0b11: 2}
        assert len(result.backend_results) == 1
        assert job.id == str(fake.execute_ref.id)

    def test_sample_without_compile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[1, 0]]])
        fake.install(monkeypatch)

        backend = QuantinuumSamplingBackend("H2-1SC", compile_on_nexus=False)
        job = backend.sample(_bell_circuit(), 1)

        assert fake.compile_calls == []
        execute_call = fake.execute_calls[0]
        assert execute_call["programs"][0].circuit is fake.uploaded[0]
        assert job.result().counts == {0b01: 1}

    def test_sample_splits_shots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0], [0, 0]], [[0, 0], [1, 1]], [[1, 1]]])
        fake.install(monkeypatch)

        backend = QuantinuumSamplingBackend("H2-1E", max_shots=2)
        job = backend.sample(_bell_circuit(), 5)

        execute_call = fake.execute_calls[0]
        assert execute_call["n_shots"] == [2, 2, 1]
        assert execute_call["programs"] == [fake.compiled_ref] * 3

        result = job.result()
        assert isinstance(result, QuantinuumSamplingResult)
        assert result.counts == {0b00: 3, 0b11: 2}
        assert len(result.backend_results) == 3

    def test_sample_applies_transpiler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)

        def transpiler(circuit: Any) -> QuantumCircuit:
            out = QuantumCircuit(circuit.qubit_count)
            out.add_X_gate(0)
            return out

        backend = QuantinuumSamplingBackend("H2-1E", circuit_transpiler=transpiler)
        backend.sample(_bell_circuit(), 1)

        ops = sorted(cmd.op.type.name for cmd in fake.uploaded[0].get_commands())
        assert ops == ["Measure", "Measure", "X"]

    def test_sample_invalid_shots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)
        backend = QuantinuumSamplingBackend("H2-1E")
        with pytest.raises(ValueError):
            backend.sample(_bell_circuit(), 0)
        assert fake.uploaded == []

    def test_sample_wraps_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)

        def failing_upload(**kwargs: Any) -> Any:
            raise RuntimeError("not logged in")

        monkeypatch.setattr(qnx.circuits, "upload", failing_upload)
        backend = QuantinuumSamplingBackend("H2-1E")
        with pytest.raises(BackendError):
            backend.sample(_bell_circuit(), 1)

    def test_result_wraps_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)

        def failing_wait(job: Any, **kwargs: Any) -> Any:
            raise RuntimeError("job errored")

        backend = QuantinuumSamplingBackend("H2-1E")
        job = backend.sample(_bell_circuit(), 1)
        monkeypatch.setattr(qnx.jobs, "wait_for", failing_wait)
        with pytest.raises(BackendError):
            job.result()

    def test_job_from_id_and_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)
        monkeypatch.setattr(qnx.jobs, "get", lambda **kwargs: fake.execute_ref)

        job = QuantinuumSamplingJob.from_id(str(fake.execute_ref.id))
        assert job.job_ref is fake.execute_ref
        job.cancel()
        assert fake.cancelled == [fake.execute_ref]

        monkeypatch.setattr(
            qnx.jobs,
            "get",
            lambda **kwargs: SimpleNamespace(id=uuid4(), job_type="compile"),
        )
        with pytest.raises(BackendError):
            QuantinuumSamplingJob.from_id("x")
