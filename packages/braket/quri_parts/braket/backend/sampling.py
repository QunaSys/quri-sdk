# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from braket.aws import AwsDevice
from braket.devices import Device
from braket.tasks import GateModelQuantumTaskResult, QuantumTask

from quri_parts.backend import (
    BackendError,
    CompositeSamplingJob,
    SamplingBackend,
    SamplingCounts,
    SamplingJob,
    SamplingResult,
)
from quri_parts.backend.qubit_mapping import BackendQubitMapping, QubitMappedSamplingJob
from quri_parts.braket.circuit import (
    BraketCircuitConverter,
    BraketTranspiler,
    convert_circuit,
)
from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler, SequentialTranspiler

from .transpiler import AwsDeviceTranspiler


class BraketSamplingResult(SamplingResult):
    """A result of a Braket sampling job."""

    def __init__(self, braket_result: GateModelQuantumTaskResult):
        if not isinstance(braket_result, GateModelQuantumTaskResult):
            raise ValueError("Only GateModelQuantumTaskResult is supported")

        self._braket_result = braket_result

    @property
    def counts(self) -> SamplingCounts:
        measurements = self._braket_result.measurements
        if measurements is None:
            raise BackendError("No valid measurement results retrieved.")
        m_qubits = self._braket_result.measured_qubits
        digits = np.array([2**q for q in m_qubits])
        return Counter(int(np.dot(digits, m)) for m in measurements)


class BraketSamplingJob(SamplingJob):
    """A job for a Braket sampling measurement."""

    def __init__(self, braket_task: QuantumTask):
        self._braket_task = braket_task

    def result(self) -> SamplingResult:
        braket_result = self._braket_task.result()
        return BraketSamplingResult(braket_result)


class BraketSamplingBackend(SamplingBackend):
    """A Braket backend for a sampling measurement.

    Args:
        device: A Braket :class:`braket.devices.Device` for circuit execution.
        circuit_converter: A function converting \
            :class:`~quri_parts.circuit.NonParametricQuantumCircuit` to \
            a Braket :class:`braket.circuits.Circuit`.
        circuit_transpiler: A transpiler applied to the circuit before running it.
            :class:`~BraketTranspiler` is used when not specified.
        enable_shots_roundup: If True, when a number of shots specified to \
            :meth:`~sample` is smaller than the minimum number of shots supported by \
            the device, it is rounded up to the minimum. In this case, it is possible \
            that shots more than specified are used. If it is strictly not allowed to \
            exceed the specified shot count, set this argument to False.
        qubit_mapping: If specified, indices of qubits in the circuit are remapped \
            before running it on the backend. It can be used when you want to use \
            specific backend qubits, e.g. those with high fidelity. \
            The mapping should be specified with "from" qubit \
            indices as keys and "to" qubit indices as values. For example, if \
            you want to map qubits 0, 1, 2, 3 to backend qubits as 0 → 4, 1 → 2, \
            2 → 5, 3 → 0, then the ``qubit_mapping`` should be \
            ``{0: 4, 1: 2, 2: 5, 3: 0}``.
        run_kwargs: Additional keyword arguments for \
            :meth:`braket.devices.Device.run` method.
    """

    def __init__(
        self,
        device: Device,
        circuit_converter: BraketCircuitConverter = convert_circuit,
        circuit_transpiler: Optional[CircuitTranspiler] = None,
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
            circuit_transpiler = BraketTranspiler()
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

        braket_circuit = self._circuit_converter(circuit, self._circuit_transpiler)
        tasks = []
        try:
            for s in shot_dist:
                tasks.append(
                    self._device.run(braket_circuit, shots=s, **self._run_kwargs)
                )
        except Exception as e:
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    # Ignore cancel errors
                    pass
            raise BackendError("Braket Device.run failed") from e

        jobs: list[SamplingJob] = [BraketSamplingJob(t) for t in tasks]
        if self._qubit_mapping is not None:
            jobs = [QubitMappedSamplingJob(job, self._qubit_mapping) for job in jobs]
        if len(jobs) == 1:
            return jobs[0]
        else:
            return CompositeSamplingJob(jobs)
