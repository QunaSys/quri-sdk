# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import Any, Optional, Sequence

import qiskit
from qiskit.providers import Job
from qiskit.providers.backend import Backend
from qiskit.result import Result

from quri_parts.backend import (
    BackendError,
    CompositeSamplingJob,
    SamplingBackend,
    SamplingCounts,
    SamplingJob,
    SamplingResult,
)
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.qiskit.circuit import QiskitCircuitConverter, convert_circuit

from .saved_sampling import (
    QiskitSavedDataSamplingJob,
    QiskitSavedDataSamplingResult,
    encode_saved_data_job_sequence_to_json,
)
from .utils import (
    convert_qiskit_sampling_count_to_qp_sampling_count,
    distribute_backend_shots,
    get_backend_min_max_shot,
    get_job_mapper_and_circuit_transpiler,
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
        return convert_qiskit_sampling_count_to_qp_sampling_count(qiskit_counts)


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
        save_data_while_sampling: If True, the circuit, n_shots and the
            sampling counts will be saved. Please use the `.jobs` or `.jobs_json`
            to access the saved data.
    """

    def __init__(
        self,
        backend: Backend,
        circuit_converter: QiskitCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
        enable_shots_roundup: bool = True,
        qubit_mapping: Optional[Mapping[int, int]] = None,
        run_kwargs: Mapping[str, Any] = {},
        save_data_while_sampling: bool = False,
    ):
        self._backend = backend

        # circuit related
        self._circuit_converter = circuit_converter

        (
            self._job_mapper,
            self._circuit_transpiler,
        ) = get_job_mapper_and_circuit_transpiler(qubit_mapping, circuit_transpiler)

        # shots related
        self._enable_shots_roundup = enable_shots_roundup
        self._min_shots, self._max_shots = get_backend_min_max_shot(backend)

        # other kwargs
        self._run_kwargs = run_kwargs

        # saving mode
        self._save_data_while_sampling = save_data_while_sampling
        self._saved_data: list[tuple[str, int, QiskitSamplingJob]] = []

    def sample(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> SamplingJob:
        if not n_shots >= 1:
            raise ValueError("n_shots should be a positive integer.")

        shot_dist = distribute_backend_shots(
            n_shots, self._min_shots, self._max_shots, self._enable_shots_roundup
        )

        qiskit_circuit = self._circuit_converter(circuit, self._circuit_transpiler)
        qiskit_circuit.measure_all()
        transpiled_circuit = qiskit.transpile(qiskit_circuit, self._backend)
        circuit_qasm_str = transpiled_circuit.qasm()

        jobs: list[QiskitSamplingJob] = []
        try:
            for s in shot_dist:
                qiskit_job = self._backend.run(
                    transpiled_circuit,
                    shots=s,
                    **self._run_kwargs,
                )
                qiskit_sampling_job = QiskitSamplingJob(qiskit_job)
                # Saving mode
                if self._save_data_while_sampling:
                    self._saved_data.append((circuit_qasm_str, s, qiskit_sampling_job))
                jobs.append(qiskit_sampling_job)

        except Exception as e:
            for qiskit_sampling_job in jobs:
                try:
                    qiskit_sampling_job._qiskit_job.cancel()
                except Exception:
                    # Ignore cancel errors
                    pass
            raise BackendError("Qiskit Device.run failed.") from e

        qubit_mapped_jobs = [self._job_mapper(job) for job in jobs]
        return (
            qubit_mapped_jobs[0]
            if len(qubit_mapped_jobs) == 1
            else CompositeSamplingJob(qubit_mapped_jobs)
        )

    @property
    def jobs(self) -> Sequence[QiskitSavedDataSamplingJob]:
        """Convert saved data to a list of QiskitSavedDataSamplingJob
        objects."""
        job_list = []
        for circuit_qasm_str, n_shots, qiskit_sampling_job in self._saved_data:
            raw_measurement_cnt = qiskit_sampling_job._qiskit_job.result().get_counts()
            saved_sampling_result = QiskitSavedDataSamplingResult(
                raw_data=raw_measurement_cnt
            )
            saved_sampling_job = QiskitSavedDataSamplingJob(
                circuit_qasm=circuit_qasm_str,
                n_shots=n_shots,
                saved_result=saved_sampling_result,
            )
            job_list.append(saved_sampling_job)
        return job_list

    @property
    def jobs_json(self) -> str:
        """Encodes the list of QiskitSavedDataSamplingJob objects to a json
        string."""
        return encode_saved_data_job_sequence_to_json(self.jobs)
