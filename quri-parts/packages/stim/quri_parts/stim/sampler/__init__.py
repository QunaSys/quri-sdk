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
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np

from quri_parts.circuit.circuit import ImmutableQuantumCircuit
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts, Sampler
from quri_parts.core.utils.concurrent import execute_concurrently

from ..circuit import convert_circuit

if TYPE_CHECKING:
    from concurrent.futures import Executor


def _sample(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
    qubit_count = circuit.qubit_count
    stim_circuit = convert_circuit(circuit)
    stim_circuit.append("M", [i for i in range(qubit_count)])
    stim_sampler = stim_circuit.compile_sampler()
    meas_results = stim_sampler.sample(shots=shots)

    pows_base_2 = np.array([2**i for i in range(qubit_count)])
    meas_results_int = np.dot(meas_results, pows_base_2).tolist()
    return Counter(meas_results_int)


def create_stim_clifford_sampler() -> Sampler:
    """Returns a :class:`~Sampler` that uses stim clifford simulator for
    sampling."""
    return _sample


def _sample_sequentially(
    _: Any, circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
    return [_sample(circuit, shots) for circuit, shots in circuit_shots_tuples]


def _sample_concurrently(
    circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]],
    executor: Optional["Executor"],
    concurrency: int = 1,
) -> Iterable[MeasurementCounts]:
    return execute_concurrently(
        _sample_sequentially, None, circuit_shots_tuples, executor, concurrency
    )


def create_stim_clifford_concurrent_sampler(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that uses stim clifford simulator
    for sampling."""

    def sampler(
        circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return _sample_concurrently(circuit_shots_tuples, executor, concurrency)

    return sampler
