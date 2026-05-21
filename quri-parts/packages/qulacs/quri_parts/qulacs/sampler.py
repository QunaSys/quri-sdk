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
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Any, Optional

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.circuit.noise import NoiseModel
from quri_parts.core.sampling import (
    ConcurrentSampler,
    GeneralSampler,
    MeasurementCounts,
    Sampler,
)
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.core.utils.concurrent import execute_concurrently

from ._backend import DEFAULT_BACKEND, QulacsBackend
from .circuit.noise import convert_circuit_with_noise_model
from .simulator import (
    create_qulacs_density_matrix_state_sampler,
    create_qulacs_ideal_density_matrix_state_sampler,
    create_qulacs_ideal_vector_state_sampler,
    create_qulacs_noisesimulator_state_sampler,
    create_qulacs_vector_state_sampler,
)
from .types import QulacsParametricStateT, QulacsStateT

if TYPE_CHECKING:
    from concurrent.futures import Executor


def create_qulacs_vector_ideal_sampler(
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> Sampler:
    """Create an ideal :class:`~Sampler` using Qulacs vector simulation.

    The returned sampler produces counts proportional to the exact
    probabilities (no shot noise).

    Args:
        backend: Backend used to construct the qulacs state and read its
            state vector.

    Returns:
        A :class:`~Sampler` returning ideal (probability-weighted) counts.
    """
    ideal_state_sampler = create_qulacs_ideal_vector_state_sampler(backend)

    def _ideal_sample(
        circuit: ImmutableQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
        return ideal_state_sampler(state, shots)

    return _ideal_sample


def create_qulacs_vector_sampler(
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> Sampler:
    """Create a :class:`~Sampler` using Qulacs vector simulation.

    Args:
        random_seed: Optional random seed for sampling.
        backend: Backend used for simulation (state construction and sampling
            strategy).

    Returns:
        A :class:`~Sampler` that samples measurement outcomes.
    """
    state_sampler = create_qulacs_vector_state_sampler(random_seed, backend)

    def _sample(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
        state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
        return state_sampler(state, shots)

    return _sample


def create_qulacs_vector_concurrent_sampler(
    random_seed: Optional[int] = None,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ConcurrentSampler:
    """Create a :class:`~ConcurrentSampler` using Qulacs vector simulation.

    Args:
        random_seed: Optional random seed for sampling.
        executor: Executor used to run samplers concurrently.
        concurrency: Maximum number of concurrent sampler invocations.
        backend: Backend used for simulation (state construction and sampling
            strategy).

    Returns:
        A :class:`~ConcurrentSampler` that samples in parallel.
    """

    def _sample_sequentially(
        _: Any,
        tuples: Sequence[tuple[ImmutableQuantumCircuit, int, Optional[int]]],
    ) -> Iterable[MeasurementCounts]:
        return [
            create_qulacs_vector_sampler(seed, backend)(circuit, shots)
            for circuit, shots, seed in tuples
        ]

    def _sample_sequentially_without_seed(
        _: Any,
        tuples: Sequence[tuple[ImmutableQuantumCircuit, int]],
    ) -> Iterable[MeasurementCounts]:
        sampler = create_qulacs_vector_sampler(None, backend)
        return [sampler(circuit, shots) for circuit, shots in tuples]

    def _concurrent_sampler(
        circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        circuit_shots_list = list(circuit_shots_tuples)
        if random_seed is None:
            return execute_concurrently(
                _sample_sequentially_without_seed,
                None,
                circuit_shots_list,
                executor,
                concurrency,
            )
        seed_counter = count(random_seed)
        seeded = [
            (circuit, shots, next(seed_counter))
            for circuit, shots in circuit_shots_list
        ]
        return execute_concurrently(
            _sample_sequentially,
            None,
            seeded,
            executor,
            concurrency,
        )

    return _concurrent_sampler


def create_qulacs_general_vector_sampler(
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> GeneralSampler[QulacsStateT, QulacsParametricStateT]:
    """Create a :class:`GeneralSampler` using Qulacs vector simulation.

    Args:
        random_seed: Optional random seed for sampling.
        backend: Backend used for simulation (state construction and sampling
            strategy).

    Returns:
        A :class:`GeneralSampler` combining a circuit sampler and a state
        sampler.
    """
    sampler = create_qulacs_vector_sampler(random_seed, backend)
    state_sampler = create_qulacs_vector_state_sampler(random_seed, backend)
    return GeneralSampler(sampler, state_sampler)


def create_qulacs_general_vector_ideal_sampler(
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> GeneralSampler[QulacsStateT, QulacsParametricStateT]:
    """Create an ideal :class:`GeneralSampler` using Qulacs vector simulation.

    Args:
        backend: Backend used to construct the qulacs state and read its
            state vector.

    Returns:
        A :class:`GeneralSampler` returning ideal (probability-weighted)
        counts.
    """
    sampler = create_qulacs_vector_ideal_sampler(backend)
    state_sampler = create_qulacs_ideal_vector_state_sampler(backend)
    return GeneralSampler(sampler, state_sampler)


def create_qulacs_stochastic_state_vector_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> Sampler:
    """Create a :class:`~Sampler` that repeats stochastic noisy state vector
    simulation once per shot.

    Args:
        model: Noise model to apply.
        random_seed: Optional random seed for sampling.
        backend: Backend used to construct the qulacs state.

    Returns:
        A :class:`~Sampler` aggregating one-shot samples from independent
        noisy trajectories.
    """

    def _sample_with_noise(
        circuit: ImmutableQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        qubit_count = circuit.qubit_count
        qs_circuit = convert_circuit_with_noise_model(circuit, model)

        sampled = []
        state = backend.init_zero_state(qubit_count)
        if random_seed is None:
            for _ in range(shots):
                state.set_computational_basis(0)
                qs_circuit.update_quantum_state(state)
                sampled += state.sampling(1)
            return Counter(sampled)

        seed_iterator = count(random_seed)
        for _ in range(shots):
            state.set_computational_basis(0)
            qs_circuit.update_quantum_state(state)
            sampled += state.sampling(1, next(seed_iterator))
        return Counter(sampled)

    return _sample_with_noise


def create_qulacs_density_matrix_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> Sampler:
    """Create a :class:`~Sampler` that samples from a noisy density matrix.

    Args:
        model: Noise model to apply.
        random_seed: Optional random seed for sampling.
        backend: Backend used to construct the qulacs density matrix.

    Returns:
        A :class:`~Sampler` sampling from the noisy density matrix.
    """
    state_sampler = create_qulacs_density_matrix_state_sampler(
        model, random_seed, backend
    )

    def _sample_with_noise(
        circuit: ImmutableQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
        return state_sampler(state, shots)

    return _sample_with_noise


def create_qulacs_density_matrix_ideal_sampler(
    model: NoiseModel,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> Sampler:
    """Create an ideal :class:`~Sampler` that returns counts proportional to
    the exact probabilities derived from a noisy density matrix.

    Args:
        model: Noise model to apply.
        backend: Backend used to construct the qulacs density matrix.

    Returns:
        A :class:`~Sampler` returning ideal (probability-weighted) counts.
    """
    ideal_state_sampler = create_qulacs_ideal_density_matrix_state_sampler(
        model, backend
    )

    def _sample_with_noise(
        circuit: ImmutableQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
        return ideal_state_sampler(state, shots)

    return _sample_with_noise


def create_qulacs_density_matrix_general_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> GeneralSampler[QulacsStateT, QulacsParametricStateT]:
    """Create a :class:`GeneralSampler` that samples from a noisy density
    matrix.

    Args:
        model: Noise model to apply.
        random_seed: Optional random seed for sampling.
        executor: Currently has no effect; reserved for future use.
        concurrency: Currently has no effect; reserved for future use.
        backend: Backend used to construct the qulacs density matrix.

    Returns:
        A :class:`GeneralSampler` combining a circuit sampler and a state
        sampler.
    """
    sampler = create_qulacs_density_matrix_sampler(model, random_seed, backend)
    state_sampler = create_qulacs_density_matrix_state_sampler(
        model, random_seed, backend
    )
    return GeneralSampler(sampler, state_sampler)


def create_qulacs_ideal_density_matrix_general_sampler(
    model: NoiseModel,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> GeneralSampler[QulacsStateT, QulacsParametricStateT]:
    """Create an ideal :class:`GeneralSampler` that returns counts proportional
    to the exact probabilities of a noisy density matrix.

    Args:
        model: Noise model to apply.
        executor: Currently has no effect; reserved for future use.
        concurrency: Currently has no effect; reserved for future use.
        backend: Backend used to construct the qulacs density matrix.

    Returns:
        A :class:`GeneralSampler` returning ideal (probability-weighted)
        counts.
    """
    sampler = create_qulacs_density_matrix_ideal_sampler(model, backend)
    state_sampler = create_qulacs_ideal_density_matrix_state_sampler(model, backend)
    return GeneralSampler(sampler, state_sampler)


def create_qulacs_noisesimulator_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> Sampler:
    """Create a :class:`~Sampler` that uses Qulacs ``NoiseSimulator``.

    Args:
        model: Noise model to apply.
        random_seed: Has no effect (``NoiseSimulator`` does not support seeding);
            a RuntimeWarning is emitted if not ``None``.
        backend: Backend used to construct the qulacs state and the
            ``NoiseSimulator`` instance.

    Returns:
        A :class:`~Sampler` sampling via Qulacs ``NoiseSimulator``.
    """
    state_sampler = create_qulacs_noisesimulator_state_sampler(
        model, random_seed, backend
    )

    def _sample_with_noise(
        circuit: ImmutableQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        return state_sampler(
            GeneralCircuitQuantumState(circuit.qubit_count, circuit), shots
        )

    return _sample_with_noise


def _create_qulacs_concurrent_sampler_with_noise_model(
    sampler_creator: Callable[[NoiseModel, Optional[int]], Sampler],
    model: NoiseModel,
    random_seed: Optional[int],
    executor: Optional["Executor"],
    concurrency: int,
) -> ConcurrentSampler:
    def _sample_sequentially(
        _: Any,
        circuit_shots_tuples: Sequence[tuple[ImmutableQuantumCircuit, int, int]],
    ) -> Iterable[MeasurementCounts]:
        return [
            sampler_creator(model, seed)(circuit, shots)
            for circuit, shots, seed in circuit_shots_tuples
        ]

    def _sample_sequentially_without_seed(
        _: Any,
        circuit_shots_tuples: Sequence[tuple[ImmutableQuantumCircuit, int]],
    ) -> Iterable[MeasurementCounts]:
        sampler = sampler_creator(model, None)
        return [sampler(circuit, shots) for circuit, shots in circuit_shots_tuples]

    def sampler(
        circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        circuit_shots_list = list(circuit_shots_tuples)
        if random_seed is None:
            return execute_concurrently(
                _sample_sequentially_without_seed,
                None,
                circuit_shots_list,
                executor,
                concurrency,
            )

        seed_counter = count(random_seed)
        seeded_circuit_shots = [
            (circuit, shots, next(seed_counter))
            for circuit, shots in circuit_shots_list
        ]

        return execute_concurrently(
            _sample_sequentially,
            None,
            seeded_circuit_shots,
            executor,
            concurrency,
        )

    return sampler


def create_qulacs_density_matrix_concurrent_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ConcurrentSampler:
    """Create a :class:`~ConcurrentSampler` that samples from a noisy density
    matrix in parallel.

    Args:
        model: Noise model to apply.
        random_seed: Optional random seed for sampling.
        executor: Executor used to run samplers concurrently.
        concurrency: Maximum number of concurrent sampler invocations.
        backend: Backend used to construct the qulacs density matrix.

    Returns:
        A :class:`~ConcurrentSampler` sampling from the noisy density matrix.
    """
    return _create_qulacs_concurrent_sampler_with_noise_model(
        partial(create_qulacs_density_matrix_sampler, backend=backend),
        model,
        random_seed,
        executor,
        concurrency,
    )


def create_qulacs_stochastic_state_vector_concurrent_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ConcurrentSampler:
    """Create a :class:`~ConcurrentSampler` that repeats stochastic noisy state
    vector simulation in parallel.

    Args:
        model: Noise model to apply.
        random_seed: Optional random seed for sampling.
        executor: Executor used to run samplers concurrently.
        concurrency: Maximum number of concurrent sampler invocations.
        backend: Backend used to construct the qulacs state.

    Returns:
        A :class:`~ConcurrentSampler` aggregating samples from independent
        noisy trajectories.
    """

    return _create_qulacs_concurrent_sampler_with_noise_model(
        partial(create_qulacs_stochastic_state_vector_sampler, backend=backend),
        model,
        random_seed,
        executor,
        concurrency,
    )


def create_qulacs_noisesimulator_concurrent_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ConcurrentSampler:
    """Create a :class:`~ConcurrentSampler` that uses Qulacs
    ``NoiseSimulator``.

    Args:
        model: Noise model to apply.
        random_seed: Has no effect (``NoiseSimulator`` does not support seeding);
            a RuntimeWarning is emitted if not ``None``.
        executor: Executor used to run samplers concurrently.
        concurrency: Maximum number of concurrent sampler invocations.
        backend: Backend used to construct the qulacs state and the
            ``NoiseSimulator`` instance.

    Returns:
        A :class:`~ConcurrentSampler` sampling via Qulacs ``NoiseSimulator``.
    """
    return _create_qulacs_concurrent_sampler_with_noise_model(
        partial(create_qulacs_noisesimulator_sampler, backend=backend),
        model,
        random_seed,
        executor,
        concurrency,
    )


def create_qulacs_noisesimulator_general_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> GeneralSampler[QulacsStateT, QulacsParametricStateT]:
    """Create a :class:`GeneralSampler` based on Qulacs ``NoiseSimulator``.

    Args:
        model: Noise model to apply.
        random_seed: Has no effect (``NoiseSimulator`` does not support seeding);
            a RuntimeWarning is emitted if not ``None``.
        executor: Currently has no effect; reserved for future use.
        concurrency: Currently has no effect; reserved for future use.
        backend: Backend used to construct the qulacs state and the
            ``NoiseSimulator`` instance.

    Returns:
        A :class:`GeneralSampler` combining a circuit sampler and a state
        sampler via ``NoiseSimulator``.
    """
    sampler = create_qulacs_noisesimulator_sampler(model, random_seed, backend)
    state_sampler = create_qulacs_noisesimulator_state_sampler(
        model, random_seed, backend
    )
    return GeneralSampler(sampler, state_sampler)
