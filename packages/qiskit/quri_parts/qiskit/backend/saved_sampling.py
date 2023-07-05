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
from collections import defaultdict
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import qiskit
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder
from qiskit.providers.backend import Backend
from typing_extensions import TypeAlias

from quri_parts.backend import (
    CompositeSamplingJob,
    SamplingBackend,
    SamplingCounts,
    SamplingJob,
    SamplingResult,
)
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.qiskit.circuit import QiskitCircuitConverter, convert_circuit

from .utils import (
    convert_qiskit_sampling_count_to_qp_sampling_count,
    distribute_backend_shots,
    get_backend_min_max_shot,
    get_job_mapper_and_circuit_transpiler,
)

SavedDataType: TypeAlias = dict[tuple[str, int], list["QiskitSavedDataSamplingJob"]]


@dataclass
class QiskitSavedDataSamplingResult(SamplingResult):
    """An object that holds a sampling count from qiskit backend output and
    converts it into quri-parts sampling count.

    The `raw_data` should take in the output of
    `qiskit_result.get_counts()`, which is a counter that uses str as
    its key.
    """

    raw_data: dict[str, int]

    @property
    def counts(self) -> SamplingCounts:
        """Convert the raw data to quri-parts sampling count.

        The quri-parts sampling count is a counter that uses int as its
        key.
        """
        return convert_qiskit_sampling_count_to_qp_sampling_count(self.raw_data)


@dataclass
class QiskitRuntimeSavedDataSamplingResult(SamplingResult):
    """An object that holds quasi disttribution and total shot count from
    qiskit runtime output and converts it into quri-parts sampling count.

    Args:
        quasi_dist: The first element of the quasi_dists attribute of a
            :class:`qiskit.primitives.SamplerResult` object.
        n_shots: The metadata[0]["shots"] output of a
            :class:`qiskit.primitives.SamplerResult` object.
    """

    quasi_dist: dict[int, float]
    n_shots: int

    @property
    def counts(self) -> SamplingCounts:
        measurements: MutableMapping[int, float] = {}
        for result, quasi_prob in self.quasi_dist.items():
            measurements[result] = quasi_prob * self.n_shots
        return measurements


@dataclass
class QiskitSavedDataSamplingJob(SamplingJob):
    """An object that represents a saved sampling job.

    Args:
        circuit_qasm: A string that represents the circuit used in a sampling job.
            Note that it should take in the qasm string of a qiskit quantum circuit.
            It can be accessed by `qiskit_circuit.qasm()`.
        n_shots: The total shots of a sampling job.
        saved_result: A `QiskitSavedDataSamplingResult` instance that represents the
            result when (circuit_str, n_shots) is passed into the sampler.
    """

    circuit_qasm: str
    n_shots: int
    saved_result: Union[
        QiskitSavedDataSamplingResult, QiskitRuntimeSavedDataSamplingResult
    ]

    def result(
        self,
    ) -> Union[QiskitSavedDataSamplingResult, QiskitRuntimeSavedDataSamplingResult]:
        return self.saved_result


class QiskitSavedDataSamplingBackend(SamplingBackend):
    """A Qiskit backend for replaying saved sampling experiments. When a
    sampler is created with a QiskitSavedDataSamplingBackend object, the
    sequence of (circuit, n_shots) pairs should be passed in to the sampler the
    same order as the orginal experiment.

    Example:
        1. Sampling experiment

        1-a: Sampling mode with data saving

        >>> backend_device = AerSimulator()
        >>> sampling_backend = QiskitSamplingBackend(
        ...     backend_device, save_data_while_sampling=True
        ... )
        >>> sampler = create_sampler_from_sampling_backend(sampling_backend)

        1-b: Perform sampling experiments

        >>> sampling_count_1 = sampler(circuit_1, n_shots_1)
        >>> sampling_count_2 = sampler(circuit_2, n_shots_2)
        >>> sampling_count_3 = sampler(circuit_3, n_shots_3)

        1-c: Dump sampling data

        >>> experiment_json_str = sampling_backend.jobs_json()

        2. Replay sampling experiment

        2-a: Create sampling backend and sampler with saved data.

        >>> saved_data_sampling_backend = QiskitSavedDataSamplingBackend(
        ...     backend = backend_device,
        ...     saved_data = experiment_json_str
        ... )

        >>> saved_data_sampler = create_sampler_from_sampling_backend(
        ...    saved_data_sampling_backend
        ... )

        2-b: Replay sampling experiment.

        (circuit, n_shots) pairs are passed in to the `saved_data_sampler`
        the same order as they were passed in to the `sampler`.

        >>> replayed_sampling_count_1 = sampler(circuit_1, n_shots_1)
        >>> replayed_sampling_count_2 = sampler(circuit_2, n_shots_2)
        >>> replayed_sampling_count_3 = sampler(circuit_3, n_shots_3)

    Args:
        backend: A Qiskit :class:`qiskit.providers.backend.Backend`
            for circuit execution.
        saved_data: A json string output by the `.json_str` property of
            `:class:`~quri_parts.qiskit.backend.QiskitSamplingBackend`.
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
        saved_data: str,
        circuit_converter: QiskitCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
        enable_shots_roundup: bool = True,
        qubit_mapping: Optional[Mapping[int, int]] = None,
        run_kwargs: Mapping[str, Any] = {},
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

        # reading mode
        self._saved_data = self._load_data(saved_data)
        self._replay_memory = {k: 0 for k in self._saved_data}

    def sample(self, circuit: NonParametricQuantumCircuit, n_shots: int) -> SamplingJob:
        if not n_shots >= 1:
            raise ValueError("n_shots should be a positive integer.")

        shot_dist = distribute_backend_shots(
            n_shots, self._min_shots, self._max_shots, self._enable_shots_roundup
        )

        qiskit_circuit = self._circuit_converter(circuit, self._circuit_transpiler)
        qiskit_circuit.measure_all()
        transpiled_circuit = qiskit.transpile(qiskit_circuit, self._backend)
        qasm_str = transpiled_circuit.qasm()
        jobs: list[SamplingJob] = []

        for s in shot_dist:
            if (key := (qasm_str, s)) in self._saved_data:
                data_position = self._replay_memory[key]
                try:
                    jobs.append(self._saved_data[key][data_position])
                    self._replay_memory[key] += 1
                except IndexError:
                    raise ValueError("Replay of this experiment is over")
            else:
                raise KeyError("This experiment is not in the saved data.")

        jobs = [self._job_mapper(job) for job in jobs]
        return jobs[0] if len(jobs) == 1 else CompositeSamplingJob(jobs)

    def _load_data(self, json_str: str) -> SavedDataType:
        saved_data = defaultdict(list)
        saved_data_seq = decode_json_to_saved_data_sequence(json_str)
        for job in saved_data_seq:
            circuit_qasm = job.circuit_qasm
            n_shots = job.n_shots
            saved_data[(circuit_qasm, n_shots)].append(job)
        return saved_data


def decode_json_to_saved_data_sequence(
    json_str: str,
) -> Sequence[QiskitSavedDataSamplingJob]:
    saved_jobs = []
    saved_data_seq = json.loads(json_str)
    for job_dict in saved_data_seq:
        saved_jobs.append(QiskitSavedDataSamplingJob(**job_dict))
    return saved_jobs


def encode_saved_data_job_sequence_to_json(
    saved_data_seq: Sequence[QiskitSavedDataSamplingJob],
) -> str:
    return json.dumps(saved_data_seq, default=pydantic_encoder)
