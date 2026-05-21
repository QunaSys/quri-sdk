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
from collections import Counter
from collections.abc import Sequence
from itertools import count
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union, overload

import qulacs as ql
from numpy import complex128, zeros
from numpy.typing import NDArray

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.circuit.noise import NoiseModel
from quri_parts.core.sampling import (
    ConcurrentStateSampler,
    MeasurementCounts,
    StateSampler,
    ideal_sample_from_density_matrix,
    ideal_sample_from_state_vector,
    sample_from_density_matrix,
    sample_from_state_vector,
)
from quri_parts.core.state import CircuitQuantumState, QuantumStateVector
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.qulacs.circuit import convert_circuit
from quri_parts.qulacs.circuit.compiled_circuit import _QulacsCircuit
from quri_parts.qulacs.circuit.noise import convert_circuit_with_noise_model

from ._backend import DEFAULT_BACKEND, QulacsBackend
from .types import QulacsStateT

if TYPE_CHECKING:
    from concurrent.futures import Executor


def _make_seed_list(length: int, random_seed: Optional[int]) -> list[Optional[int]]:
    if random_seed is None:
        return [None] * length
    seed_generator = count(random_seed)
    return [next(seed_generator) for _ in range(length)]


def _get_init_vector_from_state(state: QulacsStateT) -> NDArray[complex128]:
    n_qubits = state.qubit_count
    if isinstance(state, QuantumStateVector):
        return state.vector
    if isinstance(state, CircuitQuantumState):
        init_state_vector = zeros(2**n_qubits, dtype=complex)
        init_state_vector[0] = 1.0
        return init_state_vector
    raise TypeError(
        "the input state should be either a GeneralCircuitQuantumState\
            or a QuantumStateVector"
    )


@overload
def _evaluate_qp_state_to_qulacs_state(
    state: QulacsStateT,
    noise_model: NoiseModel,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ql.DensityMatrix:
    ...


@overload
def _evaluate_qp_state_to_qulacs_state(
    state: QulacsStateT,
    *,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ql.QuantumState:
    ...


def _evaluate_qp_state_to_qulacs_state(
    state: QulacsStateT,
    noise_model: Optional[NoiseModel] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> Union[ql.QuantumState, ql.DensityMatrix]:
    # Fast path for CircuitQuantumState (initial state is |0⟩): avoid allocating
    # a full 2**n vector on every MPI rank by delegating directly to the backend.
    # QuantumStateVector carries its own vector and is handled by the general path.
    if noise_model is None and isinstance(state, CircuitQuantumState):
        return _get_updated_qulacs_state_from_zero(state.circuit, backend)

    init_state_vector = _get_init_vector_from_state(state)
    if noise_model is None:
        return _get_updated_qulacs_state_from_vector(
            state.circuit, init_state_vector, backend
        )
    return _get_updated_qulacs_density_matrix_from_vector(
        state.circuit, init_state_vector, noise_model, backend
    )


def _get_updated_qulacs_state_from_zero(
    circuit: Union[ImmutableQuantumCircuit, _QulacsCircuit],
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ql.QuantumState:
    """Initialise to |0⟩ and apply circuit, without allocating a full 2**n
    vector.

    Uses ``backend.init_zero_state`` so that MPI backends can avoid the
    O(2**n) allocation that would otherwise happen on every rank.  Only
    valid when the logical initial state is |0⟩ (i.e. the input is a
    CircuitQuantumState).
    """
    qulacs_state = backend.init_zero_state(circuit.qubit_count)

    if isinstance(circuit, _QulacsCircuit):
        qulacs_cicuit = circuit._qulacs_circuit
    else:
        qulacs_cicuit = convert_circuit(circuit)

    qulacs_cicuit.update_quantum_state(qulacs_state)
    return qulacs_state


def _get_updated_qulacs_state_from_vector(
    circuit: Union[ImmutableQuantumCircuit, _QulacsCircuit],
    init_state: NDArray[complex128],
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ql.QuantumState:
    qulacs_state = backend.init_state(circuit.qubit_count, init_state)

    if isinstance(circuit, _QulacsCircuit):
        qulacs_cicuit = circuit._qulacs_circuit
    else:
        qulacs_cicuit = convert_circuit(circuit)

    qulacs_cicuit.update_quantum_state(qulacs_state)

    return qulacs_state


def _get_updated_qulacs_density_matrix_from_vector(
    circuit: Union[ImmutableQuantumCircuit, _QulacsCircuit],
    init_state: NDArray[complex128],
    noise_model: NoiseModel,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ql.DensityMatrix:
    qs_circuit = convert_circuit_with_noise_model(circuit, noise_model)
    density_matrix = backend.init_density_matrix(circuit.qubit_count, init_state)
    qs_circuit.update_quantum_state(density_matrix)
    return density_matrix


def _get_noise_simulator_from_vector(
    circuit: Union[ImmutableQuantumCircuit, _QulacsCircuit],
    init_state: NDArray[complex128],
    noise_model: NoiseModel,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ql.NoiseSimulator:
    """Returns a :class:`qulacs.NoiseSimulator`"""
    qs_circuit = convert_circuit_with_noise_model(circuit, noise_model)
    qs_state = backend.init_state(circuit.qubit_count, init_state)
    return backend.init_noise_simulator(qs_circuit, qs_state)


def evaluate_state_to_vector(
    state: QulacsStateT, backend: QulacsBackend = DEFAULT_BACKEND
) -> QuantumStateVector:
    """Convert GeneralCircuitQuantumState or QuantumStateVector to
    QuantumStateVector that only contains the state vector."""
    out_state_vector = _evaluate_qp_state_to_qulacs_state(state, backend=backend)
    vec = backend.get_state_vector(out_state_vector)
    return QuantumStateVector(state.qubit_count, vec)


def run_circuit(
    circuit: ImmutableQuantumCircuit,
    init_state: NDArray[complex128],
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> NDArray[complex128]:
    """Act a ImmutableQuantumCircuit onto a state vector and returns a new
    state vector."""
    qulacs_state = _get_updated_qulacs_state_from_vector(circuit, init_state, backend)
    return backend.get_state_vector(qulacs_state)


def get_marginal_probability(
    state_vector: NDArray[complex128],
    measured_values: dict[int, int],
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> float:
    """Compute the probability of obtaining a result when measuring on a subset
    of the qubits.

    state_vector:
        A 1-dimensional array representing the state vector.
    measured_values:
        A dictionary representing the desired measurement outcome on the specified
        qubtis. Suppose {0: 1, 2: 0} is passed in, it computes the probability of
        obtaining 1 on the 0th qubit and 0 on the 2nd qubit.
    """
    return backend.get_marginal_probability(state_vector, measured_values)


def create_qulacs_vector_state_sampler(
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> StateSampler[QulacsStateT]:
    """Creates a state sampler based on Qulacs circuit execution."""

    def state_sampler(state: QulacsStateT, n_shots: int) -> MeasurementCounts:
        if backend.should_use_multinomial(n_shots, state.qubit_count):
            # Use multinomial distribution for faster sampling
            state_vector = evaluate_state_to_vector(state, backend).vector
            return sample_from_state_vector(state_vector, n_shots)

        qs_state = _evaluate_qp_state_to_qulacs_state(state, backend=backend)
        if random_seed is None:
            return Counter(qs_state.sampling(n_shots))
        else:
            return Counter(qs_state.sampling(n_shots, random_seed))

    return state_sampler


def create_concurrent_vector_state_sampler(
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ConcurrentStateSampler[QulacsStateT]:
    def _sequential_vector_state_sampler(
        _: Any,
        state_shots_tuples: Sequence[tuple[QulacsStateT, int, Optional[int]]],
    ) -> Iterable[MeasurementCounts]:
        return [
            create_qulacs_vector_state_sampler(seed, backend)(state, shots)
            for state, shots, seed in state_shots_tuples
        ]

    def _sequential_vector_state_sampler_without_seed(
        _: Any, state_shots_tuples: Sequence[tuple[QulacsStateT, int]]
    ) -> Iterable[MeasurementCounts]:
        return [
            create_qulacs_vector_state_sampler(None, backend)(state, shots)
            for state, shots in state_shots_tuples
        ]

    def concurrent_state_sampler(
        state_shots_tuples: Iterable[tuple[QulacsStateT, int]]
    ) -> Iterable[MeasurementCounts]:
        state_shots_list = list(state_shots_tuples)
        if random_seed is None:
            return execute_concurrently(
                _sequential_vector_state_sampler_without_seed,
                None,
                state_shots_list,
                executor,
                concurrency,
            )

        seeds = _make_seed_list(len(state_shots_list), random_seed)

        seeded_state_shots = [
            (state, shots, seed)
            for (state, shots), seed in zip(state_shots_list, seeds)
        ]

        return execute_concurrently(
            _sequential_vector_state_sampler,
            None,
            seeded_state_shots,
            executor,
            concurrency,
        )

    return concurrent_state_sampler


def create_qulacs_ideal_vector_state_sampler(
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> StateSampler[QulacsStateT]:
    """Creates an ideal state sampler based on Qulacs circuit execution."""

    def ideal_state_sampler(
        state: Union[CircuitQuantumState, QuantumStateVector], n_shots: int
    ) -> MeasurementCounts:
        state_vector = evaluate_state_to_vector(state, backend).vector
        return ideal_sample_from_state_vector(state_vector, n_shots)

    return ideal_state_sampler


def create_qulacs_density_matrix_state_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> StateSampler[QulacsStateT]:
    """Creates a noisy state sampler for a specific noise model."""

    def density_matrix_sampler(state: QulacsStateT, shots: int) -> MeasurementCounts:
        density_matrix = _evaluate_qp_state_to_qulacs_state(
            state, model, backend=backend
        )
        qubit_count = state.qubit_count

        if shots > max(2**10, (2**qubit_count) ** 2 / 10):
            mat = density_matrix.get_matrix()
            return sample_from_density_matrix(mat, shots)

        if random_seed is None:
            return Counter(density_matrix.sampling(shots))
        else:
            return Counter(density_matrix.sampling(shots, random_seed))

    return density_matrix_sampler


def create_qulacs_ideal_density_matrix_state_sampler(
    model: NoiseModel,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> StateSampler[QulacsStateT]:
    """Creates a noisy state sampler for a specific noise model."""

    def density_matrix_sampler(state: QulacsStateT, shots: int) -> MeasurementCounts:
        density_matrix = _evaluate_qp_state_to_qulacs_state(
            state, model, backend=backend
        )
        mat = density_matrix.get_matrix()
        return ideal_sample_from_density_matrix(mat, shots)

    return density_matrix_sampler


def create_qulacs_noisesimulator_state_sampler(
    model: NoiseModel,
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> StateSampler[QulacsStateT]:
    """Returns a :class:`~ConcurrentSampler` that uses Qulacs
    NoiseSimulator."""
    if random_seed is not None:
        warnings.warn(
            "Qulacs NoiseSimulator does not support seeding. "
            "The provided random_seed is ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _noise_simulator_state_sampler(
        state: QulacsStateT, shots: int
    ) -> MeasurementCounts:
        init_vec = _get_init_vector_from_state(state)
        noise_simulator = _get_noise_simulator_from_vector(
            state.circuit,
            init_vec,
            model,
            backend,
        )
        return Counter(noise_simulator.execute(shots))

    return _noise_simulator_state_sampler
