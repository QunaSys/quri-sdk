# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
import numpy.typing as npt

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.core.sampling import (
    ConcurrentSampler,
    GeneralSampler,
    MeasurementCounts,
    Sampler,
    StateSampler,
    sample_from_probability_distribution,
)
from quri_parts.core.state import (
    GeneralCircuitQuantumState,
    ParametricQuantumStateT,
    QuantumStateT,
)
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.tensornetwork.state import convert_state

if TYPE_CHECKING:
    from concurrent.futures import Executor


def tensor_network_state_probabilities(
    circuit: ImmutableQuantumCircuit,
) -> npt.NDArray[np.float64]:
    """Returns the probabilities of the state."""
    state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
    tensor_network_state = convert_state(state)
    tensor_network_state = tensor_network_state.contract()

    tensor = np.reshape(
        tensor_network_state._container.pop().tensor,
        (2**circuit.qubit_count),
        order="F",
    )
    probabilities: npt.NDArray[np.float64] = np.abs(tensor) ** 2
    return probabilities


def tensor_network_ideal_sample(
    circuit: ImmutableQuantumCircuit, shots: int
) -> MeasurementCounts:
    """Returns the probabilities multiplied by the specific shot count."""
    probabilities = tensor_network_state_probabilities(circuit)
    samples = {b: shots * probabilities[b] for b in range(2**circuit.qubit_count)}

    return samples


def tensor_network_sample(
    circuit: ImmutableQuantumCircuit, shots: int, seed: Optional[int] = None
) -> MeasurementCounts:
    """Returns the probabilities multiplied by the specific shot count."""
    probabilities = tensor_network_state_probabilities(circuit)
    samples = sample_from_probability_distribution(shots, probabilities, seed)

    return samples


def create_tensornetwork_ideal_sampler() -> Sampler:
    """Returns a :class:`~Sampler` that uses TensorNetwork simulator for
    returning the probabilities multiplied by the specific shot count."""

    def sample(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
        return tensor_network_ideal_sample(circuit, shots)

    return sample


def create_tensornetwork_sampler(seed: Optional[int] = None) -> Sampler:
    """Returns a :class:`~Sampler` that uses TensorNetwork simulator for
    returning the samples from a probability distribution."""

    def sample(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
        return tensor_network_sample(circuit, shots, seed)

    return sample


def create_tensornetwork_state_sampler(
    seed: Optional[int] = None,
) -> StateSampler[QuantumStateT]:
    """Returns a :class:`~Sampler` that uses TensorNetwork simulator for
    returning the samples from a probability distribution."""

    def sample(state: QuantumStateT, shots: int) -> MeasurementCounts:
        return tensor_network_sample(state.circuit, shots, seed)

    return sample


def sample_concurrently(
    circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]],
    executor: Optional["Executor"],
    concurrency: int = 1,
    seed: Optional[int] = None,
) -> Iterable[MeasurementCounts]:
    def sample_sequentially(
        _: Any, circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return [
            tensor_network_sample(circuit, shots, seed)
            for circuit, shots in circuit_shots_tuples
        ]

    return execute_concurrently(
        sample_sequentially, None, circuit_shots_tuples, executor, concurrency
    )


def create_tensornetwork_concurrent_sampler(
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
    seed: Optional[int] = None,
) -> ConcurrentSampler:
    """Returns a :class:`~ConcurrentSampler` that uses TensorNetwork simulator
    for sampling.

    For now, this function works when the executor is defined like below

    Examples:
        >>> with ProcessPoolExecutor(
                max_workers=2, mp_context=get_context("spawn")
            ) as executor:
                sampler = create_tensornetwork_concurrent_sampler(
                    executor, 2
                )
                results = list(sampler([(circuit1, 1000), (circuit2, 2000)]))
    """

    def sampler(
        circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return sample_concurrently(circuit_shots_tuples, executor, concurrency, seed)

    return sampler


def create_tensornetwork_general_sampler(
    seed: Optional[int] = None,
) -> GeneralSampler[QuantumStateT, ParametricQuantumStateT]:
    return GeneralSampler(
        create_tensornetwork_sampler(seed),
        create_tensornetwork_state_sampler(seed),
    )
