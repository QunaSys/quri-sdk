# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

from quri_parts.algo.mitigation.post_selection.post_selection import (
    create_general_post_selection_concurrent_sampler,
    create_general_post_selection_sampler,
    post_selection,
)
from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit
from quri_parts.core.sampling import MeasurementCounts


def test_post_selection() -> None:
    def _filter_fn(bits: int) -> bool:
        if bits >> 2 & 1:
            return True
        return False

    meas_counts = {
        0b000: 1,
        0b001: 2,
        0b010: 3,
        0b011: 4,
        0b100: 5,
        0b101: 6,
        0b110: 7,
        0b111: 8,
    }

    assert post_selection(_filter_fn, meas_counts) == {
        0b100: 5,
        0b101: 6,
        0b110: 7,
        0b111: 8,
    }


def _mock_sampler(_: ImmutableQuantumCircuit, __: int) -> MeasurementCounts:
    return {0b01: 1, 0b10: 10, 0b111: 20, 0b0101: 5, 0b1110: 100}


def test_create_general_post_selection_sampler() -> None:
    circuit = QuantumCircuit(2)

    def _filter_fn(bits: int) -> bool:
        if bits < 3:
            return True
        return False

    ps_sampler = create_general_post_selection_sampler(_mock_sampler, _filter_fn)
    assert ps_sampler(circuit, 100) == {0b01: 1, 0b10: 10}


def _mock_concurrent_sampler(
    _: Iterable[tuple[ImmutableQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
    return [
        {0b01: 1, 0b10: 10, 0b111: 20, 0b0101: 5, 0b1110: 100},
        {0b01: 2, 0b10: 11, 0b111: 21, 0b0101: 6, 0b1110: 101},
        {0b01: 3, 0b10: 12, 0b111: 22, 0b0101: 7, 0b1110: 102},
        {0b01: 4, 0b10: 13, 0b111: 23, 0b0101: 8, 0b1110: 103},
    ]


def test_create_general_post_selection_concurrent_sampler() -> None:
    circuit = QuantumCircuit(2)

    def _filter_fn(bits: int) -> bool:
        if bits < 3:
            return True
        return False

    ps_concurrent_sampler = create_general_post_selection_concurrent_sampler(
        _mock_concurrent_sampler, _filter_fn
    )
    assert ps_concurrent_sampler(
        [(circuit, 100), (circuit, 100), (circuit, 100), (circuit, 100)]
    ) == [
        {0b01: 1, 0b10: 10},
        {0b01: 2, 0b10: 11},
        {0b01: 3, 0b10: 12},
        {0b01: 4, 0b10: 13},
    ]
