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
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

import qiskit
from qiskit.primitives import SamplerResult
from qiskit_ibm_runtime import (
    IBMBackend,
    Options,
    QiskitRuntimeService,
    RuntimeJob,
    Sampler,
    Session,
)

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


class QiskitRuntimeSamplingResult(SamplingResult):
    """Class for the results by Qiskit sampler job."""

    def __init__(self, qiskit_result: SamplerResult):
        if not isinstance(qiskit_result, SamplerResult):
            raise ValueError("Only qiskit_ibm_runtime.SamplerResult is supported")
        self._qiskit_result = qiskit_result

    @property
    def counts(self) -> SamplingCounts:
        # It must contain only one circuit result
        if len(self._qiskit_result.quasi_dists) != 1:
            raise ValueError(
                "The Result must contain distribution for one circuit but"
                f"found {len(self._qiskit_result.quasi_dists)}"
            )
        assert len(self._qiskit_result.metadata) == 1

        total_count: int = self._qiskit_result.metadata[0]["shots"]

        measurements: MutableMapping[int, float] = {}
        for result, quasi_prob in self._qiskit_result.quasi_dists[0].items():
            measurements[result] = quasi_prob * total_count
        return measurements


class QiskitRuntimeSamplingJob(SamplingJob):
    """A job for a Qiskit sampling measurement."""

    def __init__(self, qiskit_job: RuntimeJob):
        self._qiskit_job = qiskit_job

    def result(self) -> SamplingResult:
        qiskit_result: SamplerResult = self._qiskit_job.result()
        return QiskitRuntimeSamplingResult(qiskit_result)


class QiskitRuntimeSamplingBackend(SamplingBackend):
    """A Qiskit backend for a sampling measurement.

    Args:
        backend: A Qiskit :class:`qiskit_ibm_runtime.ibm_backend` that
            interfaces with IBM quantum backend.
        service: A Qiskit
        :class:`qiskit_ibm_runtime.qiskit_runtime_service.QiskitRuntimeService`
        that interacts with the Qiskit Runtime service.
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
        sampler_options: Options for providing sampler. It can be a dictionary
            or a :class:`qiskit_ibm_runtime.Options`.
        run_kwargs: Additional keyword arguments for
            :meth:`qiskit.providers.backend.Backend.run` method.
    """

    def __init__(
        self,
        backend: IBMBackend,
        service: Optional[QiskitRuntimeService] = None,
        circuit_converter: QiskitCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
        enable_shots_roundup: bool = True,
        qubit_mapping: Optional[Mapping[int, int]] = None,
        sampler_options: Union[None, Options, Dict[str, Any]] = None,
        run_kwargs: Mapping[str, Any] = {},
    ):
        self._backend = backend
        self._service = service
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
        self._max_shots: Optional[int] = getattr(backend, "max_shots", None)

        self._session = None

        if isinstance(sampler_options, dict):
            options = Options(**sampler_options)
            self._qiskit_sampler_options = options
        elif isinstance(sampler_options, Options):
            self._qiskit_sampler_options = sampler_options
        elif sampler_options is None:
            self._qiskit_sampler_options = None
        else:
            raise ValueError("Invalid type for sampler_options")

    def close(self) -> None:
        """Close the IBM session.

        This will terminate all unfinished jobs.
        """
        if self._session is None:
            raise ValueError("Session doesn't exist to close.")
        self._session.close()

    def __enter__(self) -> "QiskitRuntimeSamplingBackend":
        """The backend passed during `__init__`, is used to construct a
        session, which will be closed after the `with` scope ends."""
        session = Session(service=self._service, backend=self._backend)
        session.__enter__()

        self._session = session
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Removes the session from the scope if it exists."""
        if self._session is not None:
            self._session.__exit__(exc_type, exc_val, exc_tb)
            self._session = None

    def sample(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> SamplingJob:
        if not n_shots >= 1:
            raise ValueError("n_shots should be a positive integer.")

        if self._max_shots is not None and n_shots > self._max_shots:
            # Splitting the number of shots into pieces.
            shot_dist = [self._max_shots] * (n_shots // self._max_shots)
            remaining = n_shots % self._max_shots
            if remaining > 0:
                if remaining >= self._min_shots:
                    shot_dist.append(remaining)
                elif self._enable_shots_roundup:
                    shot_dist.append(self._min_shots)
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
        qiskit_circuit = qiskit.transpile(qiskit_circuit, self._backend)

        qiskit_jobs = []
        try:
            if self._session is None:
                # Create a session if there is no session
                with Session(service=self._service, backend=self._backend) as session:
                    runtime_sampler = Sampler(
                        session=session, options=self._qiskit_sampler_options
                    )
                    for s in shot_dist:
                        job = runtime_sampler.run(
                            qiskit_circuit, shots=s, **self._run_kwargs
                        )
                        qiskit_jobs.append(job)
            else:
                # Do not end the session if it has been already created
                runtime_sampler = Sampler(
                    session=self._session, options=self._qiskit_sampler_options
                )
                for s in shot_dist:
                    job = runtime_sampler.run(
                        qiskit_circuit, shots=s, **self._run_kwargs
                    )
                    qiskit_jobs.append(job)
        except Exception as e:
            for j in qiskit_jobs:
                try:
                    j.cancel()
                except Exception:
                    # Ignore cancel errors
                    pass
            raise BackendError("Qiskit Device run failed.") from e

        jobs: list[SamplingJob] = [QiskitRuntimeSamplingJob(j) for j in qiskit_jobs]
        if self._qubit_mapping is not None:
            jobs = [QubitMappedSamplingJob(job, self._qubit_mapping) for job in jobs]
        if len(jobs) == 1:
            return jobs[0]
        else:
            return CompositeSamplingJob(jobs)
