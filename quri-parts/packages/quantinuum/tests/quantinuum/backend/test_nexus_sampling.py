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
from qnexus.exceptions import AuthenticationError  # noqa: E402
from qnexus.models.annotations import Annotations  # noqa: E402
from qnexus.models.job_status import JobStatusEnum  # noqa: E402
from qnexus.models.references import ExecuteJobRef, ProjectRef  # noqa: E402

from quri_parts.backend import BackendError  # noqa: E402
from quri_parts.circuit import QuantumCircuit  # noqa: E402
from quri_parts.quantinuum.backend import (  # noqa: E402
    QUANTINUUM_MAX_SHOTS_PER_ITEM,
    QuantinuumSamplingBackend,
    QuantinuumSamplingJob,
    QuantinuumSamplingResult,
)
from quri_parts.quantinuum.backend.sampling import (  # noqa: E402
    _backend_result_to_sampling_counts,
    _distribute_shots,
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


def _nexus_device(
    device_name: Any,
    backend_name: str = "Quantinuum",
    nexus_hosted: bool = False,
    misc: Any = None,
) -> Any:
    info = SimpleNamespace(misc=misc) if misc is not None else None
    return SimpleNamespace(
        device_name=device_name,
        backend_name=backend_name,
        nexus_hosted=nexus_hosted,
        stored_backend_info=info,
    )


#: What qnexus.devices.get_all() returns for an account with H2 access.
FAKE_DEVICES = [
    _nexus_device("H2-1E", misc={"system_type": "emulator", "max_n_shots": 10000}),
    _nexus_device(
        "H2-1SC", misc={"system_type": "syntax checker", "max_n_shots": 10000}
    ),
    _nexus_device("H2-1", misc={"system_type": "hardware", "max_n_shots": 10000}),
    _nexus_device("H2-2", misc={"system_type": "hardware", "max_n_shots": 5000}),
    _nexus_device(
        "H2-1LE", nexus_hosted=True, misc={"system_type": "emulator", "n_shots": 10000}
    ),
    _nexus_device("Helios-1E", misc={"system_type": "emulator", "max_n_shots": 10000}),
    _nexus_device("aer_simulator", backend_name="Aer", nexus_hosted=True),
    _nexus_device(None, backend_name="Qulacs", nexus_hosted=True),
]


class FakeNexus:
    """Records calls to the qnexus functions used by the backend."""

    def __init__(self, readouts_per_item: Sequence[Sequence[Sequence[int]]]):
        self.devices: list[Any] = list(FAKE_DEVICES)
        self.device_calls = 0
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

    def get_all_devices(self, **kwargs: Any) -> list[Any]:
        self.device_calls += 1
        return self.devices

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(qnx.devices, "get_all", self.get_all_devices)
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
        assert _backend_result_to_sampling_counts(result) == {0b001: 2, 0b110: 1}

    def test_backend_result_to_sampling_counts_sorts_bits(self) -> None:
        # bits registered out of order: index 2 first
        bits = [Bit("c", 2), Bit("c", 0), Bit("c", 1)]
        result = _backend_result([[1, 0, 0]], bits)
        assert _backend_result_to_sampling_counts(result) == {0b100: 1}

    def test_distribute_shots(self) -> None:
        assert _distribute_shots(10, None) == [10]
        assert _distribute_shots(10, 10) == [10]
        assert _distribute_shots(10, 4) == [4, 4, 2]
        assert _distribute_shots(8, 4) == [4, 4]
        with pytest.raises(ValueError):
            _distribute_shots(0, None)
        with pytest.raises(ValueError):
            _distribute_shots(5, 0)


class TestQuantinuumSamplingBackend:
    def test_init_requires_device_or_config(self) -> None:
        with pytest.raises(ValueError):
            QuantinuumSamplingBackend()

        backend = QuantinuumSamplingBackend(backend_config=qnx.AerConfig())
        assert isinstance(backend.backend_config, qnx.AerConfig)

    def test_init_from_device_name(self) -> None:
        backend = QuantinuumSamplingBackend("H2-1E")
        assert isinstance(backend.backend_config, qnx.QuantinuumConfig)
        assert backend.backend_config.device_name == "H2-1E"

    def test_init_rejects_helios(self) -> None:
        with pytest.raises(ValueError, match="Helios is not supported"):
            QuantinuumSamplingBackend("Helios-1E")
        with pytest.raises(ValueError, match="Helios is not supported"):
            QuantinuumSamplingBackend(
                backend_config=qnx.HeliosConfig(system_name="Helios-1E")
            )
        # qnexus itself warns that QuantinuumConfig is deprecated for Helios
        with pytest.warns(DeprecationWarning):
            config = qnx.QuantinuumConfig(device_name="Helios-1SC")
        with pytest.raises(ValueError, match="Helios is not supported"):
            QuantinuumSamplingBackend(backend_config=config)

    def test_max_shots_default(self) -> None:
        assert QUANTINUUM_MAX_SHOTS_PER_ITEM == 10000
        # Quantinuum-hosted devices: 10000 shots per job item
        assert QuantinuumSamplingBackend("H2-1E").max_shots == 10000
        assert QuantinuumSamplingBackend("H2-2").max_shots == 10000
        # explicit value wins
        assert QuantinuumSamplingBackend("H2-1E", max_shots=5000).max_shots == 5000
        # Nexus-hosted simulators: no split
        backend = QuantinuumSamplingBackend(backend_config=qnx.AerConfig())
        assert backend.max_shots is None

    def test_max_shots_from_device_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]], [[0, 0]], [[1, 1]]])
        fake.install(monkeypatch)

        # H2-2 reports max_n_shots=5000 in the fake device list
        backend = QuantinuumSamplingBackend("H2-2")
        assert backend.max_shots == QUANTINUUM_MAX_SHOTS_PER_ITEM  # before sample
        job = backend.sample(_bell_circuit(), 12000)
        assert backend.max_shots == 5000
        assert fake.execute_calls[-1]["n_shots"] == [5000, 5000, 2000]
        assert job.result().counts == {0b00: 2, 0b11: 1}

        # an explicit value is kept
        backend = QuantinuumSamplingBackend("H2-2", max_shots=3000)
        backend.sample(_bell_circuit(), 1)
        assert backend.max_shots == 3000

        # not reported (Nexus-hosted emulator) or check skipped: fallback
        backend = QuantinuumSamplingBackend("H2-1LE")
        backend.sample(_bell_circuit(), 1)
        assert backend.max_shots == QUANTINUUM_MAX_SHOTS_PER_ITEM
        backend = QuantinuumSamplingBackend("H2-2", check_device=False)
        backend.sample(_bell_circuit(), 1)
        assert backend.max_shots == QUANTINUUM_MAX_SHOTS_PER_ITEM

    def test_sample_splits_shots_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeNexus([[[0, 0]], [[0, 0]], [[1, 1]]])
        fake.install(monkeypatch)

        backend = QuantinuumSamplingBackend("H2-1E")
        job = backend.sample(_bell_circuit(), 25000)

        execute_call = fake.execute_calls[0]
        assert execute_call["n_shots"] == [10000, 10000, 5000]
        assert len(execute_call["programs"]) == 3
        assert job.result().counts == {0b00: 2, 0b11: 1}

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

    def test_sample_checks_device_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]], [[0, 0]]])
        fake.install(monkeypatch)
        backend = QuantinuumSamplingBackend("H2-1E")
        backend.sample(_bell_circuit(), 1)
        backend.sample(_bell_circuit(), 1)
        assert fake.device_calls == 1

    def test_sample_rejects_unavailable_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)
        backend = QuantinuumSamplingBackend("H1-1E")
        with pytest.raises(BackendError) as info:
            backend.sample(_bell_circuit(), 1)
        message = str(info.value)
        assert "'H1-1E' is not available to your Nexus account" in message
        assert "H2-1, H2-1E, H2-1LE, H2-1SC, H2-2" in message
        # Helios is visible to the account but rejected by this backend,
        # so it must not be suggested
        assert "Helios" not in message
        assert "IssuerEnum.QUANTINUUM" in message
        assert fake.uploaded == []

    def test_sample_skips_device_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)
        # check_device=False: submitted even though the name is unknown
        backend = QuantinuumSamplingBackend("H1-1E", check_device=False)
        backend.sample(_bell_circuit(), 1)
        assert fake.device_calls == 0
        assert len(fake.uploaded) == 1

        # Nexus-hosted simulators are never checked
        backend = QuantinuumSamplingBackend(backend_config=qnx.AerConfig())
        backend.sample(_bell_circuit(), 1)
        assert fake.device_calls == 0

    def test_sample_proceeds_if_device_list_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)

        def broken(**kwargs: Any) -> Any:
            raise RuntimeError("devices endpoint down")

        monkeypatch.setattr(qnx.devices, "get_all", broken)
        backend = QuantinuumSamplingBackend("H1-1E")
        backend.sample(_bell_circuit(), 1)
        assert len(fake.uploaded) == 1

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

    def test_sample_checks_max_cost_list_before_upload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeNexus([[[0, 0]], [[0, 0]], [[0, 0]]])
        fake.install(monkeypatch)

        # 5 shots with max_shots=2 -> 3 items, but only 2 limits given
        backend = QuantinuumSamplingBackend(
            "H2-1E", max_shots=2, execute_kwargs={"max_cost": [1.0, 1.0]}
        )
        with pytest.raises(ValueError, match="3 job items"):
            backend.sample(_bell_circuit(), 5)
        assert fake.uploaded == []

        # one limit per item is passed through unchanged
        backend = QuantinuumSamplingBackend(
            "H2-1E", max_shots=2, execute_kwargs={"max_cost": [1.0, 1.0, 1.0]}
        )
        backend.sample(_bell_circuit(), 5)
        assert fake.execute_calls[0]["max_cost"] == [1.0, 1.0, 1.0]

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

    def test_sample_decomposes_pauli_gates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from quri_parts.circuit import gates

        fake = FakeNexus([[[0, 0]], [[0, 0]]])
        fake.install(monkeypatch)

        circuit = QuantumCircuit(2)
        circuit.add_gate(gates.Pauli([0, 1], [1, 3]))
        circuit.add_gate(gates.PauliRotation([0, 1], [1, 3], 0.3))

        # the default tket converter cannot convert these gates by itself
        backend = QuantinuumSamplingBackend("H2-1E")
        backend.sample(circuit, 1)
        ops = {cmd.op.type.name for cmd in fake.uploaded[-1].get_commands()}
        assert "Measure" in ops and len(ops) > 1

        # a user transpiler that leaves the Pauli gates alone still works
        def noop(c: Any) -> Any:
            return c

        backend = QuantinuumSamplingBackend("H2-1E", circuit_transpiler=noop)
        backend.sample(circuit, 1)
        assert len(fake.uploaded) == 2

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
            raise RuntimeError("device H9-9 does not exist")

        monkeypatch.setattr(qnx.circuits, "upload", failing_upload)
        backend = QuantinuumSamplingBackend("H2-1E")
        with pytest.raises(BackendError) as info:
            backend.sample(_bell_circuit(), 1)
        # the original error stays visible and no login advice is given
        assert "RuntimeError: device H9-9 does not exist" in str(info.value)
        assert "qnexus.login()" not in str(info.value)
        assert isinstance(info.value.__cause__, RuntimeError)

    def test_sample_not_logged_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)

        def not_authenticated(**kwargs: Any) -> Any:
            raise AuthenticationError("Not authenticated.")

        monkeypatch.setattr(qnx.devices, "get_all", not_authenticated)
        monkeypatch.setattr(qnx.projects, "get_or_create", not_authenticated)
        backend = QuantinuumSamplingBackend("H2-1E")
        with pytest.raises(BackendError) as info:
            backend.sample(_bell_circuit(), 1)
        assert "not logged in to Quantinuum Nexus" in str(info.value)
        assert "qnexus.login()" in str(info.value)
        assert isinstance(info.value.__cause__, AuthenticationError)
        assert fake.uploaded == []

        # accessing the project directly gives the same advice
        with pytest.raises(BackendError, match="qnexus.login"):
            backend.project

    def test_result_wraps_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)

        def failing_wait(job: Any, **kwargs: Any) -> Any:
            raise RuntimeError("job errored")

        backend = QuantinuumSamplingBackend("H2-1E")
        job = backend.sample(_bell_circuit(), 1)
        monkeypatch.setattr(qnx.jobs, "wait_for", failing_wait)
        with pytest.raises(BackendError) as info:
            job.result()
        assert "RuntimeError: job errored" in str(info.value)
        assert "qnexus.login()" not in str(info.value)

    def test_job_not_logged_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeNexus([[[0, 0]]])
        fake.install(monkeypatch)

        def not_authenticated(*args: Any, **kwargs: Any) -> Any:
            raise AuthenticationError("Not authenticated.")

        monkeypatch.setattr(qnx.jobs, "get", not_authenticated)
        with pytest.raises(BackendError, match="qnexus.login"):
            QuantinuumSamplingJob.from_id(str(uuid4()))

        job = QuantinuumSamplingJob(fake.execute_ref)
        monkeypatch.setattr(qnx.jobs, "status", not_authenticated)
        with pytest.raises(BackendError, match="qnexus.login"):
            job.status()
        monkeypatch.setattr(qnx.jobs, "cancel", not_authenticated)
        with pytest.raises(BackendError, match="qnexus.login"):
            job.cancel()
        monkeypatch.setattr(qnx.jobs, "wait_for", not_authenticated)
        with pytest.raises(BackendError, match="qnexus.login"):
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
