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
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias, Unpack

from quri_parts.backend import SamplingBackend
from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    UnboundParametricQuantumCircuitProtocol,
)
from quri_parts.core.operator import CommutablePauliSet, Operator
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)

#: A type variable represents *any* non-parametric quantum state classes.
#: This is different from :class:`quri_parts.core.state.QuantumStateT`;
#: ``QuantumStateT`` represents *either one of* the classes, while ``_StateT`` also
#: covers *a union of* multiple state classes.
_StateT = TypeVar("_StateT", bound=Union[CircuitQuantumState, QuantumStateVector])

#: A type variable represents *any* parametric quantum state classes.
#: This is different from :class:`quri_parts.core.state.ParametricQuantumStateT`;
#: ``ParametricQuantumStateT`` represents *either one of* the classes, while
#: ``_ParametricStateT`` also covers *a union of* multiple state classes.
_ParametricStateT = TypeVar(
    "_ParametricStateT",
    bound=Union[ParametricCircuitQuantumState, ParametricQuantumStateVector],
)

#: MeasurementCounts represents count statistics of repeated measurements of a quantum
#: circuit. Keys are observed bit patterns encoded in integers and values are counts
#: of observation of the corresponding bit patterns.
MeasurementCounts: TypeAlias = Mapping[int, Union[int, float]]

#: Sampler represents a function that samples a specified (non-parametric) circuit by
#: a specified times and returns the count statistics. In the case of an ideal Sampler,
# the return value corresponds to probabilities multiplied by shot count.
Sampler: TypeAlias = Callable[[ImmutableQuantumCircuit, int], MeasurementCounts]

#: ConcurrentSampler represents a function that samples specified (non-parametric)
#: circuits concurrently.
ConcurrentSampler: TypeAlias = Callable[
    [Iterable[tuple[ImmutableQuantumCircuit, int]]], Iterable[MeasurementCounts]
]


#: ParametricSampler represents a sampler that samples from a parametric circuit with
#: a fixed set of circuit parameters.
ParametricSampler: TypeAlias = Callable[
    [UnboundParametricQuantumCircuitProtocol, int, Sequence[float]], MeasurementCounts
]


#: ConcurrentParametricSampler represents a sampler that samples from a parametric
#: circuit with a (shot, circuit parameter) pairs.
ConcurrentParametricSampler: TypeAlias = Callable[
    [UnboundParametricQuantumCircuitProtocol, Iterable[tuple[int, Sequence[float]]]],
    Iterable[MeasurementCounts],
]


#: StateSampler representes a function that samples a specific (non-parametric) state by
#: specified times and returns the count statistics. In the case of an ideal
#: StateSampler, the return value corresponds to probabilities multiplied by shot count.
StateSampler: TypeAlias = Callable[[_StateT, int], MeasurementCounts]

#: ConcurrentSampler represents a function that samples specified (non-parametric)
#: state by specified times and returns the count statistics concurrently.
ConcurrentStateSampler: TypeAlias = Callable[
    [Iterable[tuple[_StateT, int]]], Iterable[MeasurementCounts]
]


#: ParametricStateSampler represents a state sampler that samples from a
#: parametric state with a fixed set of circuit parameters.
ParametricStateSampler: TypeAlias = Callable[
    [_ParametricStateT, int, Sequence[float]], MeasurementCounts
]


#: ConcurrentParametricStateSampler represents a state sampler that samples from a
#: parametric state with a sequence of (shot, circuit parameter) pairs.
ConcurrentParametricStateSampler: TypeAlias = Callable[
    [_ParametricStateT, Iterable[tuple[int, Sequence[float]]]],
    Iterable[MeasurementCounts],
]


@dataclass
class GeneralSampler(Generic[_StateT, _ParametricStateT]):
    sampler: Sampler
    state_sampler: StateSampler[_StateT]
    parametric_sampler: ParametricSampler = field(init=False)
    parametric_state_sampler: ParametricStateSampler[_ParametricStateT] = field(
        init=False
    )

    def __post_init__(self) -> None:
        self.parametric_sampler = create_parametric_sampler_from_sampler(self.sampler)
        self.parametric_state_sampler = (
            create_parametric_state_sampler_from_state_sampler(self.state_sampler)
        )

    @overload
    def __call__(
        self, *sampler_input: Unpack[tuple[ImmutableQuantumCircuit, int]]
    ) -> MeasurementCounts:
        """A :class:`Sampler`"""
        ...

    @overload
    def __call__(
        self,
        *sampler_input: Unpack[
            tuple[UnboundParametricQuantumCircuitProtocol, int, Sequence[float]]
        ],
    ) -> MeasurementCounts:
        """A :class:`ParametricSampler`"""

    @overload
    def __call__(
        self,
        *sampler_input: Unpack[
            tuple[
                UnboundParametricQuantumCircuitProtocol,
                Iterable[tuple[int, Sequence[float]]],
            ]
        ],
    ) -> Iterable[MeasurementCounts]:
        """A :class:`ConcurrentParametricSampler`"""
        ...

    @overload
    def __call__(
        self, *sampler_input: Unpack[tuple[_StateT, int]]
    ) -> MeasurementCounts:
        """A :class:`StateSampler`"""
        ...

    @overload
    def __call__(
        self, *sampler_input: Unpack[tuple[_ParametricStateT, int, Sequence[float]]]
    ) -> MeasurementCounts:
        """A :class:`ParametricStateSampler`"""
        ...

    @overload
    def __call__(
        self,
        *sampler_input: Unpack[
            tuple[ParametricCircuitQuantumState, int, Sequence[float]]
        ],
    ) -> MeasurementCounts:
        """A :class:`ParametricStateSampler`"""
        # NOTE: the `tuple[ParametricCircuitQuantumState, int, Sequence[float]]`
        # is added because mypy recognizes _ParametricStateT as:
        # Union[ParametricQuantumStateVector, ParametricQuantumStateVector]
        ...

    @overload
    def __call__(
        self,
        *sampler_input: Unpack[
            tuple[_ParametricStateT, Iterable[tuple[int, Sequence[float]]]]
        ],
    ) -> Iterable[MeasurementCounts]:
        """A :class:`ConcurrentParametricStateSampler`"""
        ...

    @overload
    def __call__(
        self,
        *sampler_input: Unpack[
            tuple[ParametricCircuitQuantumState, Iterable[tuple[int, Sequence[float]]]]
        ],
    ) -> Iterable[MeasurementCounts]:
        """A :class:`ConcurrentParametricStateSampler`"""
        # NOTE: the `tuple[ParametricCircuitQuantumState, int, Sequence[float]]`
        # is added because mypy recognizes _ParametricStateT as:
        # Union[ParametricQuantumStateVector, ParametricQuantumStateVector]
        ...

    @overload
    def __call__(
        self,
        *sampler_input: Union[
            tuple[ImmutableQuantumCircuit, int],
            tuple[UnboundParametricQuantumCircuitProtocol, int, Sequence[float]],
            tuple[_StateT, int],
            tuple[_ParametricStateT, int, Sequence[float]],
            tuple[ParametricCircuitQuantumState, int, Sequence[float]],
        ],
    ) -> Iterable[MeasurementCounts]:
        """Mixing :class:`ConcurrentSampler`, :class:`ConcurrentStateSampler`,
        `Iterable[tuple[UnboundParametricQuantumCircuitProtocol, int,
        Sequence[float]]]`, `Iterable[tuple[_ParametricStateT, int,
        Sequence[float]]]`"""
        # NOTE: the `tuple[ParametricCircuitQuantumState, int, Sequence[float]]`
        # is added because mypy recognizes _ParametricStateT as:
        # Union[ParametricQuantumStateVector, ParametricQuantumStateVector]
        ...

    @overload
    def __call__(
        self,
        *sampler_input: Unpack[
            tuple[
                Iterable[
                    Union[
                        tuple[ImmutableQuantumCircuit, int],
                        tuple[
                            UnboundParametricQuantumCircuitProtocol,
                            int,
                            Sequence[float],
                        ],
                        tuple[_StateT, int],
                        tuple[_ParametricStateT, int, Sequence[float]],
                        tuple[ParametricCircuitQuantumState, int, Sequence[float]],
                    ]
                ]
            ]
        ],
    ) -> Iterable[MeasurementCounts]:
        """Mixing :class:`ConcurrentSampler`, :class:`ConcurrentStateSampler`,
        `Iterable[tuple[UnboundParametricQuantumCircuitProtocol, int,
        Sequence[float]]]`, `Iterable[tuple[_ParametricStateT, int,
        Sequence[float]]]`"""
        # NOTE: the `tuple[ParametricCircuitQuantumState, int, Sequence[float]]`
        # is added because mypy recognizes _ParametricStateT as:
        # Union[ParametricQuantumStateVector, ParametricQuantumStateVector]
        ...

    def __call__(
        self, *sampler_input: Any
    ) -> Union[MeasurementCounts, Iterable[MeasurementCounts]]:
        if isinstance(sampler_input[0], ImmutableQuantumCircuit):
            return self._sample(*sampler_input)
        if isinstance(sampler_input[0], UnboundParametricQuantumCircuitProtocol):
            if isinstance(sampler_input[1], Iterable):
                return self(self._distribute_for_concurrent_sampling(*sampler_input))
            return self._parametric_sample(*sampler_input)
        if isinstance(sampler_input[0], (CircuitQuantumState, QuantumStateVector)):
            return self._sample_state(*sampler_input)
        if isinstance(
            sampler_input[0],
            (ParametricCircuitQuantumState, ParametricQuantumStateVector),
        ):
            if isinstance(sampler_input[1], Iterable):
                return self(self._distribute_for_concurrent_sampling(*sampler_input))
            return self._sample_parametric_state(*sampler_input)

        if isinstance(sampler_input[0][0], Iterable) and isinstance(
            sampler_input[0], Iterable
        ):
            return self(*sampler_input[0])

        return [cast(MeasurementCounts, self(*comb)) for comb in sampler_input]

    def _check_shot_is_int(self, shots: int) -> None:
        if not isinstance(shots, int):
            raise ValueError(
                f"Shot expected to be integer, but got {type(shots)}. "
                f"Input value is {shots}."
            )

    def _check_param_is_iterable(self, params: Sequence[float]) -> None:
        if not isinstance(params, Iterable) and not isinstance(params, np.ndarray):
            raise ValueError(
                "Circuit parameter is expected to be an iterable or an array, "
                f"but got {type(params)}. Input value is {params}."
            )

    @overload
    @staticmethod
    def _distribute_for_concurrent_sampling(
        param_circuit_or_state: UnboundParametricQuantumCircuitProtocol,
        shot_param_tuples: Iterable[tuple[int, Sequence[float]]],
    ) -> Iterable[tuple[UnboundParametricQuantumCircuitProtocol, int, Sequence[float]]]:
        ...

    @overload
    @staticmethod
    def _distribute_for_concurrent_sampling(
        param_circuit_or_state: _StateT,
        shot_param_tuples: Iterable[tuple[int, Sequence[float]]],
    ) -> Iterable[tuple[_StateT, int, Sequence[float]]]:
        ...

    @staticmethod
    def _distribute_for_concurrent_sampling(
        param_circuit_or_state: Union[UnboundParametricQuantumCircuitProtocol, _StateT],
        shot_param_tuples: Iterable[tuple[int, Sequence[float]]],
    ) -> Iterable[
        tuple[
            Union[UnboundParametricQuantumCircuitProtocol, _StateT],
            int,
            Sequence[float],
        ]
    ]:
        return [
            (param_circuit_or_state, shot, param) for shot, param in shot_param_tuples
        ]

    def _sample(
        self, circuit: ImmutableQuantumCircuit, shots: int
    ) -> MeasurementCounts:
        self._check_shot_is_int(shots)
        return self.sampler(circuit, shots)

    def _parametric_sample(
        self,
        param_circuit: UnboundParametricQuantumCircuitProtocol,
        shots: int,
        params: Sequence[float],
    ) -> MeasurementCounts:
        self._check_shot_is_int(shots)
        self._check_param_is_iterable(params)
        return self.parametric_sampler(param_circuit, shots, params)

    def _sample_state(self, state: _StateT, shots: int) -> MeasurementCounts:
        self._check_shot_is_int(shots)
        return self.state_sampler(state, shots)

    def _sample_parametric_state(
        self,
        param_state: _ParametricStateT,
        shots: int,
        params: Sequence[float],
    ) -> MeasurementCounts:
        self._check_shot_is_int(shots)
        self._check_param_is_iterable(params)
        return self.parametric_state_sampler(param_state, shots, params)


def create_parametric_sampler_from_sampler(sampler: Sampler) -> ParametricSampler:
    """Create a :class:`ParametricSampler` from a :class:`Sampler`."""

    def _parametric_sampler(
        param_circuit: UnboundParametricQuantumCircuitProtocol,
        measurement_cnt: int,
        param: Sequence[float],
    ) -> MeasurementCounts:
        bound_circuit = param_circuit.bind_parameters(param)
        return sampler(bound_circuit, measurement_cnt)

    return _parametric_sampler


def create_concurrent_parametric_sampler_from_concurrent_sampler(
    concurrent_sampler: ConcurrentSampler,
) -> ConcurrentParametricSampler:
    """Create a :class:`ConcurrentParametricSampler` from a
    :class:`ConcurrentSampler`."""

    def _concurrent_parametric_sampler(
        param_circuit: UnboundParametricQuantumCircuitProtocol,
        shot_param_tuples: Iterable[tuple[int, Sequence[float]]],
    ) -> Iterable[MeasurementCounts]:
        circuit_shot_tuples = [
            (param_circuit.bind_parameters(param), shot)
            for shot, param in shot_param_tuples
        ]
        return concurrent_sampler(circuit_shot_tuples)

    return _concurrent_parametric_sampler


def create_parametric_state_sampler_from_state_sampler(
    state_sampler: StateSampler[_StateT],
) -> ParametricStateSampler[_ParametricStateT]:
    """Create a :class:`ParametricStateSampler` from a
    :class:`StateSampler`."""

    def _parametric_state_sampler(
        param_state: _ParametricStateT,
        measurement_cnt: int,
        param: Sequence[float],
    ) -> MeasurementCounts:
        bound_state = cast(_StateT, param_state.bind_parameters(param))
        return state_sampler(bound_state, measurement_cnt)

    return _parametric_state_sampler


def create_concurrent_parametric_state_sampler_from_concurrent_state_sampler(
    concurrent_state_sampler: ConcurrentStateSampler[_StateT],
) -> ConcurrentParametricStateSampler[_ParametricStateT]:
    """Create a :class:`ConcurrentParametricStateSampler` from a
    :class:`ConcurrentStateSampler`.
    """

    def _concurrent_parametric_state_sampler(
        param_state: _ParametricStateT,
        shot_param_tuples: Iterable[tuple[int, Sequence[float]]],
    ) -> Iterable[MeasurementCounts]:
        state_shot_tuples = [
            (cast(_StateT, param_state.bind_parameters(param)), shot)
            for shot, param in shot_param_tuples
        ]
        return concurrent_state_sampler(state_shot_tuples)

    return _concurrent_parametric_state_sampler


def sample_from_probability_distribution(
    n_sample: int,
    probability_distribution: Union[Sequence[float], npt.NDArray[np.float64]],
    seed: Optional[int] = None,
) -> MeasurementCounts:
    """Sample from a probibility distribution."""
    rng = np.random.default_rng(seed=seed)
    rounded_prob = np.round(probability_distribution, 12)
    norm = np.sum(rounded_prob)
    assert np.isclose(norm, 1.0), "Probabilty does not sum to 1.0"
    rounded_prob = rounded_prob / norm
    counts = rng.multinomial(n_sample, rounded_prob)
    return Counter(dict(((i, count) for i, count in enumerate(counts) if count > 0)))


def sample_from_state_vector(
    state_vector: npt.NDArray[np.complex128], n_shots: int
) -> MeasurementCounts:
    """Perform sampling from a state vector."""
    n_qubits: float = np.log2(state_vector.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."
    if not np.isclose(np.linalg.norm(state_vector), 1):
        raise ValueError("probabilities do not sum to 1")
    probs = cast(npt.NDArray[np.float64], np.abs(state_vector) ** 2)
    return sample_from_probability_distribution(n_shots, probs)


def ideal_sample_from_state_vector(
    state_vector: npt.NDArray[np.complex128], n_shots: int
) -> MeasurementCounts:
    """Perform ideal sampling from a state vector."""
    n_qubits: float = np.log2(state_vector.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."
    if not np.isclose(np.linalg.norm(state_vector), 1):
        raise ValueError("probabilities do not sum to 1")

    probs = np.abs(state_vector) ** 2
    return {i: prob * n_shots for i, prob in enumerate(probs)}


def sample_from_density_matrix(
    density_matrix: npt.NDArray[np.complex128], n_shots: int
) -> MeasurementCounts:
    assert (
        density_matrix.ndim == 2
    ), f"Density matrix must be 2 dimensional, but got {density_matrix.ndim}."

    n_qubits: float = np.log2(density_matrix.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."

    if not np.isclose(np.trace(density_matrix), 1):
        raise ValueError("probabilities do not sum to 1")

    probs = np.diag(density_matrix).real
    return sample_from_probability_distribution(n_shots, probs)


def ideal_sample_from_density_matrix(
    density_matrix: npt.NDArray[np.complex128], n_shots: int
) -> MeasurementCounts:
    assert (
        density_matrix.ndim == 2
    ), f"Density matrix must be 2 dimensional, but got {density_matrix.ndim}."

    n_qubits: float = np.log2(density_matrix.shape[0])
    assert n_qubits.is_integer(), "Length of the state vector must be a power of 2."

    if not np.isclose(np.trace(density_matrix), 1):
        raise ValueError("probabilities do not sum to 1")

    probs = np.diag(density_matrix).real
    return {i: prob * n_shots for i, prob in enumerate(probs)}


def create_sampler_from_sampling_backend(backend: SamplingBackend) -> Sampler:
    """Create a simple :class:`~Sampler` using a :class:`~SamplingBackend`."""

    def sampler(circuit: ImmutableQuantumCircuit, n_shots: int) -> MeasurementCounts:
        job = backend.sample(circuit, n_shots)
        return job.result().counts

    return sampler


def create_concurrent_sampler_from_sampling_backend(
    backend: SamplingBackend,
) -> ConcurrentSampler:
    """Create a simple :class:`~ConcurrentSampler` using a
    :class:`~SamplingBackend`."""

    def sampler(
        shot_circuit_pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        jobs = [
            backend.sample(circuit, n_shots) for circuit, n_shots in shot_circuit_pairs
        ]
        return map(lambda j: j.result().counts, jobs)

    return sampler


def create_sampler_from_concurrent_sampler(
    concurrent_sampler: ConcurrentSampler,
) -> Sampler:
    def sampler(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
        return next(iter(concurrent_sampler([(circuit, shots)])))

    return sampler


class PauliSamplingSetting(NamedTuple):
    pauli_set: CommutablePauliSet
    n_shots: int


#: PauliSamplingShotsAllocator represents a function that distributes
#: a given number of sampling shots to each :class:`~CommutablePauliSet`.
PauliSamplingShotsAllocator: TypeAlias = Callable[
    [Operator, Collection[CommutablePauliSet], int], Collection[PauliSamplingSetting]
]


#: WeightedSamplingShotsAllocator represents a function that distributes
#: a given number of sampling shots based on a set of weights.
WeightedSamplingShotsAllocator: TypeAlias = Callable[
    [Sequence[complex], int], Sequence[int]
]


__all__ = [
    "MeasurementCounts",
    "Sampler",
    "ConcurrentSampler",
    "ParametricSampler",
    "ConcurrentParametricSampler",
    "StateSampler",
    "ConcurrentStateSampler",
    "ParametricStateSampler",
    "ConcurrentParametricStateSampler",
    "GeneralSampler",
    "create_parametric_sampler_from_sampler",
    "create_concurrent_parametric_sampler_from_concurrent_sampler",
    "create_parametric_state_sampler_from_state_sampler",
    "create_concurrent_parametric_state_sampler_from_concurrent_state_sampler",
    "sample_from_probability_distribution",
    "sample_from_state_vector",
    "ideal_sample_from_state_vector",
    "sample_from_density_matrix",
    "ideal_sample_from_density_matrix",
    "create_sampler_from_sampling_backend",
    "create_concurrent_sampler_from_sampling_backend",
    "create_sampler_from_concurrent_sampler",
    "PauliSamplingSetting",
    "PauliSamplingShotsAllocator",
    "WeightedSamplingShotsAllocator",
]
