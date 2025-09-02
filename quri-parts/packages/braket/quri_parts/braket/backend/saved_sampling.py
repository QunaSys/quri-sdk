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

from braket.aws import AwsDevice
from braket.devices import Device
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder
from typing_extensions import TypeAlias

from quri_parts.backend import (
    CompositeSamplingJob,
    SamplingBackend,
    SamplingCounts,
    SamplingJob,
    SamplingResult,
)
from quri_parts.backend.qubit_mapping import BackendQubitMapping, QubitMappedSamplingJob
from quri_parts.braket.circuit import (
    BraketCircuitConverter,
    BraketSetTranspiler,
    convert_circuit,
)
from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler, SequentialTranspiler

from .transpiler import AwsDeviceTranspiler

SavedDataType: TypeAlias = dict[tuple[str, int], list["BraketSavedDataSamplingJob"]]


@dataclass
class BraketSavedDataSamplingResult(SamplingResult):
    """An object that holds a sampling count from Braket backend output and
    converts it into quri-parts sampling count.

    The `raw_data` should take in the output of
    `braket_task.result().measurement_counts`, which is a counter that
    uses str as its key.
    """

    raw_data: dict[str, int]

    @property
    def counts(self) -> SamplingCounts:
        """Convert the raw data to quri-parts sampling count.

        The quri-parts sampling count is a counter that uses int as its
        key.
        """
        measurements: MutableMapping[int, int] = {}
        for result in self.raw_data:
            measurements[int(result[::-1], 2)] = self.raw_data[result]
        return measurements


@dataclass
class BraketSavedDataSamplingJob(SamplingJob):
    """An object that represents a saved sampling job.

    Args:
        circuit_program_str: A string that represents the circuit used in a sampling
            job. Note that it should take in the program string of a braket quantum
            circuit. It can be accessed by `braket_circuit.to_ir().json()`.
        n_shots: The total shots of a sampling job.
        saved_result: A `BraketSavedDataSamplingResult` instance that represents the
            result when (circuit_str, n_shots) is passed into the sampler.
    """

    circuit_program_str: str
    n_shots: int
    saved_result: BraketSavedDataSamplingResult

    def result(self) -> Union[BraketSavedDataSamplingResult]:
        return self.saved_result


class BraketSavedDataSamplingBackend(SamplingBackend):
    """A Braket backend for replaying saved sampling experiments. When a
    sampler is created with a BraketSavedDataSamplingBackend object, the
    sequence of (circuit, n_shots) pairs should be passed in to the sampler the
    same order as the orginal experiment.

    Example:
        1. Sampling experiment

        1-a: Sampling mode with data saving

        >>> backend_device = LocalSimulator()
        >>> sampling_backend = BraketSamplingBackend(
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

        >>> saved_data_sampling_backend = BraketSavedDataSamplingBackend(
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
        device: A Braket :class:`braket.devices.Device` for circuit execution.
        saved_data: A json string output by the `.jobs_json` property of
            `:class:`~quri_parts.braket.backend.BraketSamplingBackend`.
        circuit_converter: A function converting
            :class:`~quri_parts.circuit.ImmutableQuantumCircuit` to
            a Braket :class:`braket.circuits.Circuit`.
        circuit_transpiler: A transpiler applied to the circuit before running it.
            :class:`~BraketSetTranspiler` is used when not specified.
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
            :meth:`braket.devices.Device.run` method.
    """

    def __init__(
        self,
        device: Device,
        saved_data: str,
        circuit_converter: BraketCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = BraketSetTranspiler(),
        enable_shots_roundup: bool = True,
        qubit_mapping: Optional[Mapping[int, int]] = None,
        run_kwargs: Mapping[str, Any] = {},
    ):
        self._device = device
        self._circuit_converter = circuit_converter

        self._qubit_mapping = None
        if qubit_mapping is not None:
            self._qubit_mapping = BackendQubitMapping(qubit_mapping)

        if circuit_transpiler is None:
            circuit_transpiler = BraketSetTranspiler()
        if isinstance(device, AwsDevice):
            circuit_transpiler = SequentialTranspiler(
                [circuit_transpiler, AwsDeviceTranspiler(device)]
            )
        if self._qubit_mapping:
            circuit_transpiler = SequentialTranspiler(
                [circuit_transpiler, self._qubit_mapping.circuit_transpiler]
            )
        self._circuit_transpiler = circuit_transpiler

        self._enable_shots_roundup = enable_shots_roundup
        self._run_kwargs = run_kwargs

        self._min_shots = 1
        self._max_shots: Optional[int] = None
        if isinstance(device, AwsDevice):
            min, max = device.properties.service.shotsRange
            if min > 0:
                self._min_shots = min
            if max > 0:
                self._max_shots = max

        # saving mode
        self._saved_data = self._load_data(saved_data)
        self._replay_memory = {k: 0 for k in self._saved_data}

    def sample(self, circuit: ImmutableQuantumCircuit, n_shots: int) -> SamplingJob:
        if not n_shots >= 1:
            raise ValueError("n_shots should be a positive integer.")
        if self._max_shots is not None and n_shots > self._max_shots:
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

        braket_circuit = self._circuit_converter(circuit, self._circuit_transpiler)
        program_str = braket_circuit.to_ir().json()
        jobs: list[SamplingJob] = []

        for s in shot_dist:
            if (key := (program_str, s)) in self._saved_data:
                data_position = self._replay_memory[key]
                try:
                    jobs.append(self._saved_data[key][data_position])
                    self._replay_memory[key] += 1
                except IndexError:
                    raise ValueError("Replay of this experiment is over")
            else:
                raise KeyError("This experiment is not in the saved data.")

        if self._qubit_mapping is not None:
            jobs = [QubitMappedSamplingJob(job, self._qubit_mapping) for job in jobs]
        return jobs[0] if len(jobs) == 1 else CompositeSamplingJob(jobs)

    def _load_data(self, json_str: str) -> SavedDataType:
        saved_data = defaultdict(list)
        saved_data_seq = decode_json_to_saved_data_sequence(json_str)
        for job in saved_data_seq:
            circuit_program = job.circuit_program_str
            n_shots = job.n_shots
            saved_data[(circuit_program, n_shots)].append(job)
        return saved_data


def decode_json_to_saved_data_sequence(
    json_str: str,
) -> Sequence[BraketSavedDataSamplingJob]:
    saved_jobs = []
    saved_data_seq = json.loads(json_str)
    for job_dict in saved_data_seq:
        saved_jobs.append(BraketSavedDataSamplingJob(**job_dict))
    return saved_jobs


def encode_saved_data_job_sequence_to_json(
    saved_data_seq: Sequence[BraketSavedDataSamplingJob],
) -> str:
    return json.dumps(saved_data_seq, default=pydantic_encoder)
