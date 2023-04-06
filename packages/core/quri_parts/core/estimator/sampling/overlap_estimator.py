# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from math import sqrt

from quri_parts.circuit import inverse_circuit
from quri_parts.core.estimator import (
    Estimate,
    OverlapEstimator,
    OverlapWeightedSumEstimator,
)
from quri_parts.core.sampling import (
    ConcurrentSampler,
    MeasurementCounts,
    WeightedSamplingShotsAllocator,
)
from quri_parts.core.state import CircuitQuantumState


class _OverlapEstimate:
    def __init__(self, sampling_counts: MeasurementCounts) -> None:
        self._sampling_counts = sampling_counts
        self._total_counts = sum(sampling_counts.values())

    @cached_property
    def value(self) -> float:
        val = self._sampling_counts[0] / self._total_counts
        return val

    @cached_property
    def error(self) -> float:
        p = self._sampling_counts[0] / self._total_counts
        # Variance of the binomial distribution can be estimated as
        var = self._total_counts * p * (1 - p)
        std_err = sqrt(var) / self._total_counts
        return std_err


class _OverlapWeightedSumEstimate:
    def __init__(
        self, overlap_estimates: Sequence[Estimate[float]], weights: Sequence[complex]
    ):
        self._overlap_estimates = overlap_estimates
        self._weights = weights

    @cached_property
    def value(self) -> complex:
        val = sum(
            weight * overlap.value
            for weight, overlap in zip(self._weights, self._overlap_estimates)
        )
        return val

    @cached_property
    def error(self) -> float:
        square_err = sum(
            abs(weight) ** 2 * estimates.error**2
            for weight, estimates in zip(self._weights, self._overlap_estimates)
        )
        return sqrt(square_err)


@dataclass
class _ConstOverlapEstimate:
    value: complex
    error: float = 0.0


def sampling_overlap_estimate(
    ket: CircuitQuantumState,
    bra: CircuitQuantumState,
    shots: int,
    sampler: ConcurrentSampler,
) -> Estimate[float]:
    """Estimate the magnitude squared overlap of a pair of quantum states by
    sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        ket: The state whose direct circuit is used
        bra: The state whose inverse circuit is used


    Returns:
        The estimated value (can be accessed with :attr:`.value`) with standard error
            of estimation (can be accessed with :attr:`.error`).
    """

    circuit_and_shots = (
        ket.circuit + inverse_circuit(bra.circuit),
        shots,
    )
    counts = next(iter(sampler([circuit_and_shots])))

    return _OverlapEstimate(counts)


def create_sampling_overlap_estimator(
    shots: int,
    sampler: ConcurrentSampler,
) -> OverlapEstimator[CircuitQuantumState]:
    """Create a :class:`OverlapEstimator` that estimates the squared magnitude
    of the overlap of a pair of wavefunctions by sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        shots: Number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
    """

    def estimator(
        ket: CircuitQuantumState,
        bra: CircuitQuantumState,
    ) -> Estimate[float]:
        return sampling_overlap_estimate(ket, bra, shots, sampler)

    return estimator


def sampling_overlap_weighted_sum_estimate(
    kets: Sequence[CircuitQuantumState],
    bras: Sequence[CircuitQuantumState],
    weights: Sequence[complex],
    total_shots: int,
    sampler: ConcurrentSampler,
    shots_allocator: WeightedSamplingShotsAllocator,
) -> Estimate[complex]:
    """Estimate the weighted sum of magnitude squared overlaps of pairs of
    quantum states by a sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        kets: The states whose direct circuit is used
        bras: The states whose inverse circuit is used
        weights: Weights used in summing the overlaps
        total_shots: Total number of shots used for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        shots_allocator: shot allocator which takes weights as an argument.

    Returns:
        The estimated value (can be accessed with :attr:`.value`) with standard error
            of estimation (can be accessed with :attr:`.error`).
    """
    len_kets = len(kets)
    len_bras = len(bras)
    len_weights = len(weights)

    if any(len_kets != x for x in {len_bras, len_weights}):
        raise ValueError(
            f"Number of kets ({len_kets}) and "
            f"number of bras ({len_bras}) and "
            f"number of weights ({len_weights}) does not match."
        )

    if len_kets == 0:
        raise ValueError("Empty input sequences are not supported.")

    shot_counts = shots_allocator(weights, total_shots)

    circuits_and_shots = [
        (ket.circuit + inverse_circuit(bra.circuit), shot_count)
        for ket, bra, shot_count in zip(kets, bras, shot_counts)
    ]

    estimates = [_OverlapEstimate(estimate) for estimate in sampler(circuits_and_shots)]

    return _OverlapWeightedSumEstimate(estimates, weights)


def create_sampling_overlap_weighted_sum_estimator(
    total_shots: int,
    sampler: ConcurrentSampler,
    shots_allocator: WeightedSamplingShotsAllocator,
) -> OverlapWeightedSumEstimator[CircuitQuantumState]:
    """Create a :class:`OverlapWeightedSumEstimator` that estimates the squared
    magnitude of the overlap of pairs of wavefunctions by sampling measurement
    and returns a weighted sum.

    The sampling measurements are configured with arguments as follows.

    Args:
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        shots_allocator: shot allocator which takes weights as an argument.
    """

    def estimator(
        kets: Sequence[CircuitQuantumState],
        bras: Sequence[CircuitQuantumState],
        weights: Sequence[complex],
    ) -> Estimate[complex]:
        return sampling_overlap_weighted_sum_estimate(
            kets, bras, weights, total_shots, sampler, shots_allocator
        )

    return estimator
