# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from types import TracebackType
from typing import Any, Dict, Optional, Sequence, Type, Union

import qiskit
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import qasm3
from qiskit.primitives import PrimitiveResult
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService
from qiskit_ibm_runtime import RuntimeJobV2 as RuntimeJob
from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Session

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
from .tracker import Tracker
from .utils import (
    distribute_backend_shots,
    get_backend_min_max_shot,
    get_job_mapper_and_circuit_transpiler,
)

RUNTIME_BACKEND_MAX_EXE_TIME = 300.0


class QiskitRuntimeSamplingResult(SamplingResult):
    """Class for the results by Qiskit sampler job."""

    def __init__(self, qiskit_result: PrimitiveResult):
        if not isinstance(qiskit_result, PrimitiveResult):
            raise ValueError("Only qiskit_ibm_runtime.PrimitiveResult is supported")
        self._check_only_1_result(qiskit_result)
        self._qiskit_result = qiskit_result

    def _check_only_1_result(self, qiskit_result: PrimitiveResult) -> None:
        pub_res_list = qiskit_result._pub_results
        if len(pub_res_list) != 1:
            raise ValueError(
                "The Result must contain distribution for one circuit but"
                f"found {len(pub_res_list)}"
            )

    @property
    def counts(self) -> SamplingCounts:
        measurement_cnt_str: MutableMapping[str, int] = self._qiskit_result[
            0
        ].data.meas.get_counts()
        measurement_cnt = {int(b, 2): c for b, c in measurement_cnt_str.items()}
        return measurement_cnt


class QiskitRuntimeSamplingJob(SamplingJob):
    """A job for a Qiskit sampling measurement."""

    def __init__(self, qiskit_job: RuntimeJob):
        self._qiskit_job = qiskit_job

    def result(self) -> SamplingResult:
        qiskit_result = self._qiskit_job.result()
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
        save_data_while_sampling: If True, the circuit, n_shots and the
            sampling counts will be saved. Please use the `.jobs` or `.jobs_json`
            to access the saved data.
        total_time_limit: The total time limit the jobs submitted by this backend
            can use.

            - A :class:`~Tracker` is created when the time limit is set. The tracker can be accessed by the .tracker attribute.

            - If the job execution time exceeds the time limit, new call to the :meth:`~sample` will be rejected and all current running jobs will be cancelled.
        single_job_max_execution_time:
            Maximum execution time limit of a submitted (circuit, n_shot) pair. The value should be greater or equal to 300 seconds.
        strict_time_limit:
            The Runtime backend can only abort jobs that has been executed longer than 300 seconds or above.
            If this option is set to True, the sampling backend will reject jobs whose time limit setting is less than 300 seconds.
    """  # noqa:

    def __init__(
        self,
        backend: IBMBackend,
        service: Optional[QiskitRuntimeService] = None,
        circuit_converter: QiskitCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
        enable_shots_roundup: bool = True,
        qubit_mapping: Optional[Mapping[int, int]] = None,
        sampler_options: Union[None, SamplerOptions, Dict[str, Any]] = None,
        run_kwargs: Mapping[str, Any] = {},
        save_data_while_sampling: bool = False,
        total_time_limit: Optional[int] = None,
        single_job_max_execution_time: Optional[int] = None,
        strict_time_limit: bool = False,
    ):
        self._backend = backend
        self._service = service
        self._session = None

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
        self._saved_data: list[tuple[str, int, QiskitRuntimeSamplingJob]] = []

        # Sampler options
        if isinstance(sampler_options, dict):
            options = SamplerOptions(**sampler_options)
            self._qiskit_sampler_options = options
        elif isinstance(sampler_options, SamplerOptions):
            self._qiskit_sampler_options = sampler_options
        elif sampler_options is None:
            self._qiskit_sampler_options = None
        else:
            raise ValueError("Invalid type for sampler_options")

        # Tracker related
        self.tracker: Optional[Tracker] = None
        self._time_limit = 0.0
        self._single_job_max_execution_time = single_job_max_execution_time
        self._strict = strict_time_limit

        if total_time_limit is not None:
            self.tracker = Tracker()
            self._time_limit = total_time_limit

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

    def _run_tracker(self) -> None:
        assert self.tracker is not None
        if self.tracker.total_run_time >= self._time_limit:
            for job in self.tracker.running_jobs:
                job._qiskit_job.cancel()
            raise RuntimeError(
                "The submission of this job is aborted due to run time limit of "
                f"{self._time_limit} seconds is exceeded. Other unfinished jobs "
                "are also aborted."
            )

    def _check_execution_time_limitability(
        self, batch_exe_time: Optional[float], batch_time_left: Optional[float]
    ) -> None:
        if (
            batch_time_left is not None
            and batch_time_left < RUNTIME_BACKEND_MAX_EXE_TIME
        ):
            if self._strict:
                raise BackendError(
                    f"Max execution time limit of {batch_time_left} "
                    "seconds cannot be followed strictly."
                )
            else:
                warnings.warn(
                    f"The time limit of {batch_time_left} seconds "
                    "is likely going to be exceeded."
                )

        if batch_exe_time is not None and batch_exe_time < RUNTIME_BACKEND_MAX_EXE_TIME:
            if self._strict:
                raise BackendError(
                    f"Max execution time limit of {batch_exe_time} "
                    "seconds cannot be followed strictly."
                )
            else:
                warnings.warn(
                    f"The time limit of {batch_exe_time} seconds "
                    "is likely going to be exceeded."
                )

    def _get_sampler_option_with_time_limit(
        self, batch_exe_time: Optional[int], batch_time_left: Optional[int]
    ) -> SamplerOptions:
        options = (
            deepcopy(self._qiskit_sampler_options)
            if self._qiskit_sampler_options is not None
            else SamplerOptions()
        )

        if batch_exe_time is not None and batch_time_left is not None:
            options.max_execution_time = max(
                RUNTIME_BACKEND_MAX_EXE_TIME, min(batch_time_left, batch_exe_time)
            )

        elif batch_exe_time is not None:
            options.max_execution_time = max(
                RUNTIME_BACKEND_MAX_EXE_TIME, batch_exe_time
            )

        elif batch_time_left is not None:
            options.max_execution_time = max(
                RUNTIME_BACKEND_MAX_EXE_TIME, batch_time_left
            )

        return options

    def _get_batch_execution_time_and_time_left(
        self, shot_dist: Sequence[int]
    ) -> tuple[Optional[int], Optional[int]]:
        n_batch = len(shot_dist)
        batch_execution_time, batch_time_left = None, None

        if self._single_job_max_execution_time is not None:
            batch_execution_time = int(self._single_job_max_execution_time // n_batch)

        if self.tracker is not None:
            time_left = self._time_limit - self.tracker.total_run_time
            batch_time_left = int(time_left // n_batch)

        return batch_execution_time, batch_time_left

    def _execute_shots(
        self,
        runtime_sampler: Sampler,
        qiskit_circuit: QiskitQuantumCircuit,
        shot_dist: Sequence[int],
        jobs_list: list[QiskitRuntimeSamplingJob],
    ) -> None:
        circuit_qasm_str = qasm3.dumps(qiskit_circuit)
        for s in shot_dist:
            if self.tracker is not None:
                self._run_tracker()

            qiskit_runtime_job = runtime_sampler.run(
                [qiskit_circuit], shots=s, **self._run_kwargs
            )
            print(qiskit_runtime_job)
            qiskit_runtime_sampling_job = QiskitRuntimeSamplingJob(qiskit_runtime_job)

            if self.tracker is not None:
                self.tracker.add_job_for_tracking(qiskit_runtime_sampling_job)

            # Saving mode
            if self._save_data_while_sampling:
                self._saved_data.append(
                    (circuit_qasm_str, s, qiskit_runtime_sampling_job)
                )
            jobs_list.append(qiskit_runtime_sampling_job)

    def sample(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> SamplingJob:
        if not n_shots >= 1:
            raise ValueError("n_shots should be a positive integer.")

        # Distribute shot count and execution time
        shot_dist = distribute_backend_shots(
            n_shots, self._min_shots, self._max_shots, self._enable_shots_roundup
        )
        (
            single_batch_execution_time,
            single_batch_time_left,
        ) = self._get_batch_execution_time_and_time_left(shot_dist)

        self._check_execution_time_limitability(
            single_batch_execution_time, single_batch_time_left
        )

        qiskit_sampler_options = self._get_sampler_option_with_time_limit(
            single_batch_execution_time, single_batch_time_left
        )

        # Convert and transpile circuits
        qiskit_circuit = self._circuit_converter(circuit, self._circuit_transpiler)
        qiskit_circuit.measure_all()
        transpiled_circuit = qiskit.transpile(qiskit_circuit, self._backend)

        jobs: list[QiskitRuntimeSamplingJob] = []
        try:
            if self._session is None:
                # Create a session if there is no session
                with Session(service=self._service, backend=self._backend) as session:
                    runtime_sampler = Sampler(
                        session=session, options=qiskit_sampler_options
                    )
                    self._execute_shots(
                        runtime_sampler, transpiled_circuit, shot_dist, jobs
                    )

            else:
                # Do not end the session if it has been already created
                runtime_sampler = Sampler(
                    session=self._session, options=qiskit_sampler_options
                )
                self._execute_shots(
                    runtime_sampler, transpiled_circuit, shot_dist, jobs
                )

        except Exception as e:
            for qiskit_runtime_sampling_job in jobs:
                try:
                    qiskit_runtime_sampling_job._qiskit_job.cancel()
                except Exception:
                    # Ignore cancel errors
                    pass
            raise BackendError(f"Qiskit Device run failed. Failed reason:\n{e}") from e

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
        for circuit_qasm_str, n_shots, qiskit_runtime_sampling_job in self._saved_data:
            result = qiskit_runtime_sampling_job._qiskit_job.result()
            saved_sampling_result = QiskitSavedDataSamplingResult(
                result[0].data.meas.get_counts()
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
