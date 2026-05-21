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
from numpy import complex128
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


def _init_qulacs_state(
    state: QulacsStateT, backend: QulacsBackend = DEFAULT_BACKEND
) -> ql.QuantumState:
    """Construct a qulacs QuantumState from a quri-parts state via ``backend``.

    Dispatches by state type:

    - :class:`CircuitQuantumState`: initialised to |0⟩ via
      :meth:`QulacsBackend.init_zero_state`.
    - :class:`QuantumStateVector`: built from its stored vector via
      :meth:`QulacsBackend.init_state`.
    """
    if isinstance(state, CircuitQuantumState):
        return backend.init_zero_state(state.qubit_count)
    if isinstance(state, QuantumStateVector):
        return backend.init_state(state.qubit_count, state.vector)
    raise TypeError(
        "the input state should be either a CircuitQuantumState "
        "or a QuantumStateVector"
    )


def _init_qulacs_density_matrix(
    state: QulacsStateT, backend: QulacsBackend = DEFAULT_BACKEND
) -> ql.DensityMatrix:
    """Construct a qulacs DensityMatrix from a quri-parts state via
    ``backend``.

    Dispatches by state type:

    - :class:`CircuitQuantumState`: initialised to |0⟩⟨0| via
      :meth:`QulacsBackend.init_zero_density_matrix`.
    - :class:`QuantumStateVector`: built from its stored vector via
      :meth:`QulacsBackend.init_density_matrix`.
    """
    if isinstance(state, CircuitQuantumState):
        return backend.init_zero_density_matrix(state.qubit_count)
    if isinstance(state, QuantumStateVector):
        return backend.init_density_matrix(state.qubit_count, state.vector)
    raise TypeError(
        "the input state should be either a CircuitQuantumState "
        "or a QuantumStateVector"
    )


def _apply_circuit_to_qulacs_state(
    circuit: Union[ImmutableQuantumCircuit, _QulacsCircuit],
    qulacs_state: Union[ql.QuantumState, ql.DensityMatrix],
) -> None:
    """Apply ``circuit`` in-place to ``qulacs_state``."""
    if isinstance(circuit, _QulacsCircuit):
        qulacs_circuit = circuit._qulacs_circuit
    else:
        qulacs_circuit = convert_circuit(circuit)
    qulacs_circuit.update_quantum_state(qulacs_state)


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
    """Convert a quri-parts state to a qulacs state with the circuit applied.

    Returns a :class:`ql.DensityMatrix` if ``noise_model`` is given, otherwise a
    :class:`ql.QuantumState`.
    """
    if noise_model is None:
        qulacs_state = _init_qulacs_state(state, backend)
        _apply_circuit_to_qulacs_state(state.circuit, qulacs_state)
        return qulacs_state

    density_matrix = _init_qulacs_density_matrix(state, backend)
    qs_circuit = convert_circuit_with_noise_model(state.circuit, noise_model)
    qs_circuit.update_quantum_state(density_matrix)
    return density_matrix


def _get_updated_qulacs_state_from_vector(
    circuit: Union[ImmutableQuantumCircuit, _QulacsCircuit],
    init_state: NDArray[complex128],
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> ql.QuantumState:
    """Initialise a state from ``init_state`` and apply ``circuit``."""
    qulacs_state = backend.init_state(circuit.qubit_count, init_state)
    _apply_circuit_to_qulacs_state(circuit, qulacs_state)
    return qulacs_state


def evaluate_state_to_vector(
    state: QulacsStateT, backend: QulacsBackend = DEFAULT_BACKEND
) -> QuantumStateVector:
    """Apply ``state``'s circuit to its initial vector and return the resulting
    state vector.

    Args:
        state: A quri-parts state whose circuit will be applied to its
            initial vector.
        backend: Backend used to construct the qulacs state and read its
            state vector.

    Returns:
        A :class:`QuantumStateVector` holding the resulting amplitudes.
    """
    out_state_vector = _evaluate_qp_state_to_qulacs_state(state, backend=backend)
    vec = backend.get_state_vector(out_state_vector)
    return QuantumStateVector(state.qubit_count, vec)


def run_circuit(
    circuit: ImmutableQuantumCircuit,
    init_state: NDArray[complex128],
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> NDArray[complex128]:
    """Apply ``circuit`` to ``init_state`` and return the resulting state
    vector.

    Args:
        circuit: Circuit to apply.
        init_state: Initial state vector.
        backend: Backend used to construct the qulacs state and read its
            state vector.

    Returns:
        The state vector after applying ``circuit`` to ``init_state``.
    """
    qulacs_state = _get_updated_qulacs_state_from_vector(circuit, init_state, backend)
    return backend.get_state_vector(qulacs_state)


def get_marginal_probability(
    state_vector: NDArray[complex128],
    measured_values: dict[int, int],
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> float:
    """Compute the marginal probability of a partial measurement outcome.

    Args:
        state_vector: A 1-dimensional array representing the state vector.
        measured_values: Desired measurement outcomes keyed by qubit index.
            For example, ``{0: 1, 2: 0}`` requests the probability of
            obtaining 1 on qubit 0 and 0 on qubit 2.
        backend: Backend used to compute the marginal probability.

    Returns:
        The marginal probability of the requested outcome.
    """
    return backend.get_marginal_probability(state_vector, measured_values)


def create_qulacs_vector_state_sampler(
    random_seed: Optional[int] = None,
    backend: QulacsBackend = DEFAULT_BACKEND,
) -> StateSampler[QulacsStateT]:
    """Create a state sampler based on Qulacs vector simulation.

    Args:
        random_seed: Optional random seed for sampling.
        backend: Backend used for simulation (state construction and sampling
            strategy).

    Returns:
        A :class:`StateSampler` that samples measurement outcomes.
    """

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
    """Create a concurrent state sampler based on Qulacs vector simulation.

    Args:
        executor: Executor used to run samplers concurrently.
        concurrency: Maximum number of concurrent sampler invocations.
        random_seed: Optional random seed for sampling.
        backend: Backend used for simulation (state construction and sampling
            strategy).

    Returns:
        A :class:`ConcurrentStateSampler` that samples in parallel.
    """

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
    """Create an ideal state sampler returning shot counts proportional to
    exact probabilities.

    Args:
        backend: Backend used to construct the qulacs state and read its
            state vector.

    Returns:
        A :class:`StateSampler` returning ideal (probability-weighted) counts.
    """

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
    """Create a noisy state sampler using a density matrix.

    Args:
        model: Noise model to apply.
        random_seed: Optional random seed for sampling.
        backend: Backend used to construct the qulacs density matrix.

    Returns:
        A :class:`StateSampler` sampling from the noisy density matrix.
    """

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
    """Create an ideal noisy state sampler using a density matrix.

    Args:
        model: Noise model to apply.
        backend: Backend used to construct the qulacs density matrix.

    Returns:
        A :class:`StateSampler` returning ideal (probability-weighted) counts
        from the noisy density matrix.
    """

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
    """Create a state sampler that uses Qulacs ``NoiseSimulator``.

    Args:
        model: Noise model to apply.
        random_seed: Has no effect (``NoiseSimulator`` does not support seeding);
            a RuntimeWarning is emitted if not ``None``.
        backend: Backend used to construct the qulacs state and the
            ``NoiseSimulator`` instance.

    Returns:
        A :class:`StateSampler` sampling via Qulacs ``NoiseSimulator``.
    """
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
        qs_state = _init_qulacs_state(state, backend)
        qs_circuit = convert_circuit_with_noise_model(state.circuit, model)
        noise_simulator = backend.init_noise_simulator(qs_circuit, qs_state)
        return Counter(noise_simulator.execute(shots))

    return _noise_simulator_state_sampler
