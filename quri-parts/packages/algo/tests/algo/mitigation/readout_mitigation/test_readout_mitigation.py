# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from quri_parts.algo.mitigation.readout_mitigation import (
    create_filter_matrix,
    create_readout_mitigation_concurrent_sampler,
    create_readout_mitigation_sampler,
    readout_mitigation,
)
from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit
from quri_parts.core.sampling import MeasurementCounts

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


_count_matrix = [
    [989100, 2780, 2030, 2550, 1320, 1920, 190, 110],
    [1460, 991542, 2783, 1650, 1220, 225, 850, 270],
    [7518, 14552, 976928, 844, 114, 21, 16, 7],
    [4590, 3956, 3251, 984943, 2115, 1111, 29, 5],
    [5, 305, 540, 225, 998716, 184, 16, 9],
    [2770, 5996, 1859, 5141, 2605, 971234, 2495, 7900],
    [9530, 5670, 4746, 17483, 2560, 5440, 952517, 2054],
    [410, 366, 486, 114, 424, 530, 156, 997514],
]


_noisy_vector = [114, 1225, 3560, 977850, 1607, 2430, 5963, 7251]


qubit_count = 3
dim = 2**qubit_count
shots = 1000000


class MockRepeatConcurrentSampler:
    def __init__(self) -> None:
        self._counter = 0

    def __call__(
        self, pairs: Iterable[tuple[ImmutableQuantumCircuit, int]]
    ) -> Iterable[MeasurementCounts]:
        cs = []
        for _ in pairs:
            cs.append({i: _count_matrix[self._counter][i] for i in range(dim)})
            self._counter = (self._counter + 1) % dim
        return cs


def test_readout_mitigation() -> None:
    filter_matrix = create_filter_matrix(
        qubit_count=qubit_count, sampler=MockRepeatConcurrentSampler(), shots=shots
    )
    assert np.allclose(
        np.identity(dim),
        np.matmul(np.array(_count_matrix).T, filter_matrix) / 1.0e6,
    )

    noisy_counts = {i: _noisy_vector[i] for i in range(dim)}
    ideal_counts = list(readout_mitigation([noisy_counts], filter_matrix))[0]

    ideal_array: "npt.NDArray[np.int_]" = np.array(
        [ideal_counts[k] if k in ideal_counts else 0.0 for k in range(dim)]
    )

    ic = filter_matrix.dot(np.array(_noisy_vector))
    assert np.array_equal(ideal_array, np.where(ic > 0.0, ic, 0.0))


def test_create_sampler() -> None:
    dummy_circuit = QuantumCircuit(qubit_count)

    sampler = MockRepeatConcurrentSampler()
    filter_matrix = create_filter_matrix(
        qubit_count=qubit_count, sampler=sampler, shots=shots
    )
    counts = readout_mitigation(sampler([(dummy_circuit, shots)]), filter_matrix)

    wrapped_sampler = create_readout_mitigation_sampler(
        qubit_count=qubit_count, sampler=MockRepeatConcurrentSampler(), shots=shots
    )
    wrapped_counts = wrapped_sampler(dummy_circuit, shots)

    assert next(iter(counts)) == wrapped_counts


def test_create_concurrent_sampler() -> None:
    dummy_circuit = QuantumCircuit(qubit_count)

    sampler = MockRepeatConcurrentSampler()
    filter_matrix = create_filter_matrix(
        qubit_count=qubit_count, sampler=sampler, shots=shots
    )
    counts = readout_mitigation(sampler([(dummy_circuit, shots)]), filter_matrix)

    wrapped_sampler = create_readout_mitigation_concurrent_sampler(
        qubit_count=qubit_count, sampler=MockRepeatConcurrentSampler(), shots=shots
    )
    wrapped_counts = wrapped_sampler([(dummy_circuit, shots)])

    assert list(counts) == list(wrapped_counts)
