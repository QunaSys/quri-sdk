# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Iterable

import pytest

from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit
from quri_parts.core.sampling import (
    ConcurrentSampler,
    MeasurementCounts,
    create_readout_error_concurrent_sampler,
)


def fixed_sampler(counts: MeasurementCounts) -> ConcurrentSampler:
    def sampler(
        pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        return [counts for _ in pairs]

    return sampler


def sample(
    counts: MeasurementCounts,
    qubit_count: int,
    p0_to_1: float | list[float],
    p1_to_0: float | list[float],
    seed: int | None = 0,
) -> MeasurementCounts:
    sampler = create_readout_error_concurrent_sampler(
        fixed_sampler(counts), p0_to_1, p1_to_0, seed
    )
    shots = int(sum(counts.values()))
    return list(sampler([(QuantumCircuit(qubit_count), shots)]))[0]


def test_zero_error_keeps_counts() -> None:
    counts = {0b0101: 30, 0b0011: 70}
    assert sample(counts, 4, 0.0, 0.0) == counts


def test_certain_error_flips_every_bit() -> None:
    assert sample({0b0101: 30, 0b0011: 70}, 4, 1.0, 1.0) == {0b1010: 30, 0b1100: 70}


def test_asymmetric_error_flips_only_ones() -> None:
    assert sample({0b0101: 30, 0b0011: 70}, 4, 0.0, 1.0) == {0: 100}


def test_per_qubit_probabilities() -> None:
    assert sample({0b000: 10}, 3, [0.0, 1.0, 0.0], 0.0) == {0b010: 10}


def test_flip_rates_match_probabilities() -> None:
    shots = 100000
    noisy = sample({0b01: shots}, 2, 0.1, 0.2)
    # Qubit 0 reads 1 and qubit 1 reads 0 before readout errors.
    p_qubit0_flipped = sum(c for bits, c in noisy.items() if not bits & 0b01) / shots
    p_qubit1_flipped = sum(c for bits, c in noisy.items() if bits & 0b10) / shots
    assert p_qubit0_flipped == pytest.approx(0.2, abs=0.01)
    assert p_qubit1_flipped == pytest.approx(0.1, abs=0.01)
    assert sum(noisy.values()) == shots


def test_same_seed_gives_same_counts() -> None:
    counts = {0b0101: 500, 0b0011: 500}
    assert sample(counts, 4, 0.1, 0.1, seed=3) == sample(counts, 4, 0.1, 0.1, seed=3)


def test_supports_64_qubits() -> None:
    bits = (1 << 63) | 1
    assert sample({bits: 5}, 64, 0.0, 1.0) == {0: 5}


def test_rejects_non_integer_counts() -> None:
    with pytest.raises(ValueError, match="integer counts"):
        sample({0: 0.5, 1: 0.5}, 1, 0.1, 0.1)


@pytest.mark.parametrize("bad", [1.5, -0.1, math.nan, [0.1, math.nan]])
def test_rejects_invalid_probability(bad: float | list[float]) -> None:
    for p0_to_1, p1_to_0 in [(bad, 0.1), (0.1, bad)]:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sample({0: 10}, 2, p0_to_1, p1_to_0)


def test_rejects_probability_length_mismatch() -> None:
    cases: list[tuple[float | list[float], float | list[float]]] = [
        ([0.1] * 3, 0.1),
        (0.1, [0.1] * 3),
    ]
    for p0_to_1, p1_to_0 in cases:
        with pytest.raises(ValueError, match="Expected 2 .* got 3"):
            sample({0: 10}, 2, p0_to_1, p1_to_0)
