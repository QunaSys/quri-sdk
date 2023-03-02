# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping, MutableMapping
from typing import Any, Optional

from qiskit.providers import Job
from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.result import Result

from quri_parts.backend import (
    BackendError,
    CompositeSamplingJob,
    SamplingBackend,
    SamplingCounts,
    SamplingJob,
    SamplingResult,
)
from quri_parts.backend.qubit_mapping import BackendQubitMapping, QubitMappedSamplingJob
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler, SequentialTranspiler
from quri_parts.qiskit.circuit import (
    QiskitCircuitConverter,
    QiskitTranspiler,
    convert_circuit,
)


class QiskitSamplingResult(SamplingResult):
    """A result of a Qiskit sampling job."""

    def __init__(self, qiskit_result: Result):
        if not isinstance(qiskit_result, Result):
            raise ValueError("Only qiskit.result.Result is supported")
        self._qiskit_result = qiskit_result

    @property
    def counts(self) -> SamplingCounts:
        qiskit_counts = self._qiskit_result.get_counts()
        measurements: MutableMapping[int, int] = {}
        for result in qiskit_counts:
            measurements[int(result, 2)] = qiskit_counts[result]
        return measurements


class QiskitSamplingJob(SamplingJob):
    """A job for a Qiskit sampling measurement."""

    def __init__(self, qiskit_job: Job):
        self._qiskit_job = qiskit_job

    def result(self) -> SamplingResult:
        qiskit_result: Result = self._qiskit_job.result()
        return QiskitSamplingResult(qiskit_result)


class QiskitSamplingBackend(SamplingBackend):
    """A Qiskit backend for a sampling measurement.

    Args:
        backend: A Qiskit :class:`qiskit.providers.backend.Backend`
            for circuit execution.
        circuit_converter: A function converting
            :class:`~quri_parts.circuit.NonParametricQuantumCircuit` to
            a Qiskit :class:`qiskit.circuit.QuantumCircuit`.
        circuit_transpiler: A transpiler applied to the circuit before running it.
            :class:`~QiskitTranspiler` is used when not specified.
        enable_shots_roundup: If True, when a number of shots specified to
            :meth:`~sample` is smaller than the minimum number of shots supported by
            the device, it is rounded up to the minimum. In this case, it is possible
            that shots more than specified are used. If it is strictly not allowed to
            exceed the specified shot count, set this argument to False.
        qubit_mapping: If specified, indices of qubits in the circuit are remapped
            before running it on the backend. It can be used when you want to use
            specific backend qubits, e.g. those with high fidelity.
            The mapping should be specified with "from" qubit
            indices as keys and "to" qubit indices as values. For example, if
            you want to map qubits 0, 1, 2, 3 to backend qubits as 0 → 4, 1 → 2,
            2 → 5, 3 → 0, then the ``qubit_mapping`` should be
            ``{0: 4, 1: 2, 2: 5, 3: 0}``.
        run_kwargs: Additional keyword arguments for
            :meth:`qiskit.providers.backend.Backend.run` method.
    """

    def __init__(
        self,
        backend: Backend,
        circuit_converter: QiskitCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
        enable_shots_roundup: bool = True,
        qubit_mapping: Optional[Mapping[int, int]] = None,
        run_kwargs: Mapping[str, Any] = {},
    ):
        self._backend = backend
        self._circuit_converter = circuit_converter

        self._qubit_mapping = None
        if qubit_mapping is not None:
            self._qubit_mapping = BackendQubitMapping(qubit_mapping)

        if circuit_transpiler is None:
            circuit_transpiler = QiskitTranspiler()
        if self._qubit_mapping:
            circuit_transpiler = SequentialTranspiler(
                [circuit_transpiler, self._qubit_mapping.circuit_transpiler]
            )
        self._circuit_transpiler = circuit_transpiler

        self._enable_shots_roundup = enable_shots_roundup
        self._run_kwargs = run_kwargs

        self._min_shots = 1
        self._max_shots: Optional[int] = None
        if isinstance(backend, BackendV1):
            max_shots = backend.configuration().max_shots
            if max_shots > 0:
                self._max_shots = max_shots

        if not isinstance(backend, (BackendV1, BackendV2)):
            raise BackendError("Backend not supported.")

    def sample(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> SamplingJob:
        if not n_shots >= 1:
            raise ValueError("n_shots should be a positive integer.")
        if self._max_shots is not None and n_shots > self._max_shots:
            shot_dist = [self._max_shots] * (n_shots // self._max_shots)
            remaining = n_shots % self._max_shots
            if remaining >= self._min_shots or self._enable_shots_roundup:
                shot_dist.append(max(remaining, self._min_shots))
        else:
            if n_shots >= self._min_shots or self._enable_shots_roundup:
                shot_dist = [max(n_shots, self._min_shots)]
            else:
                raise ValueError(
                    f"n_shots is smaller than minimum shot count ({self._min_shots}) "
                    "supported by the device. Try larger n_shots or use "
                    "enable_shots_roundup=True when creating the backend."
                )

        qiskit_circuit = self._circuit_converter(circuit, self._circuit_transpiler)
        qiskit_circuit.measure_all()
        qiskit_jobs = []
        try:
            for s in shot_dist:
                qiskit_jobs.append(
                    self._backend.run(qiskit_circuit, shots=s, **self._run_kwargs)
                )
        except Exception as e:
            for j in qiskit_jobs:
                try:
                    j.cancel()
                except Exception:
                    # Ignore cancel errors
                    pass
            raise BackendError("Qiskit Device.run failed.") from e

        jobs: list[SamplingJob] = [QiskitSamplingJob(j) for j in qiskit_jobs]
        if self._qubit_mapping is not None:
            jobs = [QubitMappedSamplingJob(job, self._qubit_mapping) for job in jobs]
        if len(jobs) == 1:
            return jobs[0]
        else:
            return CompositeSamplingJob(jobs)
