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
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import qulacs
from numpy.random import default_rng

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.noise import NoiseModel
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts, Sampler
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.core.utils.concurrent import execute_concurrently

from .circuit.noise import convert_circuit_with_noise_model
from .simulator import (
    create_qulacs_ideal_vector_state_sampler,
    create_qulacs_vector_state_sampler,
)

if TYPE_CHECKING:
    from concurrent.futures import Executor

_state_vector_sampler = create_qulacs_vector_state_sampler()
_ideal_vector_sampler = create_qulacs_ideal_vector_state_sampler()


def _sample(circuit: NonParametricQuantumCircuit, shots: int) -> MeasurementCounts:
    state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
    return _state_vector_sampler(state, shots)


def _ideal_sample(
    circuit: NonParametricQuantumCircuit, shots: int
) -> MeasurementCounts:
    state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
    return _ideal_vector_sampler(state, shots)


def create_qulacs_vector_ideal_sampler() -> Sampler:
    """Returns a :class:`~Sampler` that uses Qulacs vector simulator for
    returning the probabilities multiplied by the specific shot count."""
    return _ideal_sample


def create_qulacs_vector_sampler() -> Sampler:
    """Returns a :class:`~Sampler` that uses Qulacs vector simulator for
    sampling."""
    return _sample


def _sample_sequentially(
    _: Any, circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
    return [_sample(circuit, shots) for circuit, shots in circuit_shots_tuples]


def _sample_concurrently(
    circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]],
    executor: Optional["Executor"],
    concurrency: int = 1,
) -> Iterable[MeasurementCounts]:
    return execute_concurrently(
        _sample_sequentially, None, circuit_shots_tuples, executor, concurrency
    )


def create_qulacs_vector_concurrent_sampler(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that uses Qulacs vector simulator
    for sampling."""

    def sampler(
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return _sample_concurrently(circuit_shots_tuples, executor, concurrency)

    return sampler


def create_qulacs_stochastic_state_vector_sampler(model: NoiseModel) -> Sampler:
    """Returns a :class:`~Sampler` that repeats Qulacs state vector simulation
    for shot times with noise model."""

    def _sample_with_noise(
        circuit: NonParametricQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        qubit_count = circuit.qubit_count
        qs_circuit = convert_circuit_with_noise_model(circuit, model)

        sampled = []
        state = qulacs.QuantumState(qubit_count)
        for _ in range(shots):
            state.set_computational_basis(0)
            qs_circuit.update_quantum_state(state)
            sampled += state.sampling(1)
        return Counter(sampled)

    return _sample_with_noise


def create_qulacs_density_matrix_sampler(model: NoiseModel) -> Sampler:
    """Returns a :class:`~Sampler` that uses Qulacs simulator using density
    matrix with noise model."""

    def _sample_with_noise(
        circuit: NonParametricQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        qubit_count = circuit.qubit_count
        qs_circuit = convert_circuit_with_noise_model(circuit, model)
        state = qulacs.DensityMatrix(qubit_count)
        qs_circuit.update_quantum_state(state)

        if shots > max(2**10, (2**qubit_count) ** 2 / 10):
            mat = state.get_matrix()
            probs = [mat[i, i].real for i in range(2**qubit_count)]
            rng = default_rng()
            counts = rng.multinomial(shots, probs)
            return dict((i, count) for i, count in enumerate(counts) if count > 0)
        else:
            return Counter(state.sampling(shots))

    return _sample_with_noise


def create_qulacs_density_matrix_ideal_sampler(model: NoiseModel) -> Sampler:
    """Returns a :class:`~Sampler` that uses Qulacs simulator using density
    matrix with noise model and provides the exact probabilities multiplied
    with the shot count."""

    def _sample_with_noise(
        circuit: NonParametricQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        qubit_count = circuit.qubit_count
        qs_circuit = convert_circuit_with_noise_model(circuit, model)
        state = qulacs.DensityMatrix(qubit_count)
        qs_circuit.update_quantum_state(state)

        probs = list(np.abs(np.diag(state.get_matrix())))  # type: ignore
        return {i: prob * shots for i, prob in enumerate(probs)}

    return _sample_with_noise


def create_qulacs_noisesimulator_sampler(model: NoiseModel) -> Sampler:
    """Returns a :class:`~Sampler` that uses Qulacs NoiseSimulator."""

    def _sample_with_noise(
        circuit: NonParametricQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        qubit_count = circuit.qubit_count
        qs_circuit = convert_circuit_with_noise_model(circuit, model)
        state = qulacs.QuantumState(qubit_count)
        sim = qulacs.NoiseSimulator(qs_circuit, state)

        return Counter(sim.execute(shots))

    return _sample_with_noise


def _create_qulacs_concurrent_sampler_with_noise_model(
    sampler_creator: Callable[[NoiseModel], Sampler],
    model: NoiseModel,
    executor: Optional["Executor"],
    concurrency: int,
) -> ConcurrentSampler:
    noise_sampler = sampler_creator(model)

    def _sample_sequentially(
        _: Any, circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return [
            noise_sampler(circuit, shots) for circuit, shots in circuit_shots_tuples
        ]

    def sampler(
        circuit_shots_tuples: Iterable[tuple[NonParametricQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return execute_concurrently(
            _sample_sequentially, None, circuit_shots_tuples, executor, concurrency
        )

    return sampler


def create_qulacs_density_matrix_concurrent_sampler(
    model: NoiseModel,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that uses Qulacs simulator using
    density matrix with noise model for sampling."""

    return _create_qulacs_concurrent_sampler_with_noise_model(
        create_qulacs_density_matrix_sampler,
        model,
        executor,
        concurrency,
    )


def create_qulacs_stochastic_state_vector_concurrent_sampler(
    model: NoiseModel,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that repeats Qulacs state vector
    simulation for shot times with noise model."""

    return _create_qulacs_concurrent_sampler_with_noise_model(
        create_qulacs_stochastic_state_vector_sampler,
        model,
        executor,
        concurrency,
    )


def create_qulacs_noisesimulator_concurrent_sampler(
    model: NoiseModel,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that uses Qulacs
    NoiseSimulator."""

    return _create_qulacs_concurrent_sampler_with_noise_model(
        create_qulacs_noisesimulator_sampler,
        model,
        executor,
        concurrency,
    )
