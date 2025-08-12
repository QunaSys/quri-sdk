# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock, patch

import numpy as np
import pytest

from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.sampling import PauliSamplingSetting
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
    create_proportional_shots_allocator,
    create_weighted_random_shots_allocator,
)

PAULI_LABELS = [
    pauli_label("Z0"),
    pauli_label("X1 Y2"),
    pauli_label("X1 Z3"),
    PAULI_IDENTITY,
    pauli_label("X2 Z4"),
]
COEFS = [0.2, 1.5, -2.0j, 1.0, -0.5 + 1.2j]


@pytest.fixture
def operator() -> Operator:
    op = Operator()
    for p_label, coef in zip(PAULI_LABELS, COEFS):
        op[p_label] = coef
    return op


def test_equipartition_individual(operator: Operator) -> None:
    groups = {
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1]}),
        frozenset({PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    }
    total_shots = 1111
    allocator = create_equipartition_shots_allocator()
    shots_allocation = allocator(operator, groups, total_shots)
    assert len(shots_allocation) == 4
    shots_per_term_expected = total_shots // 4
    for setting in shots_allocation:
        assert setting.n_shots == shots_per_term_expected


def test_equipartition_grouped(operator: Operator) -> None:
    groups = {
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    }
    total_shots = 1111
    allocator = create_equipartition_shots_allocator()
    shots_allocation = allocator(operator, groups, total_shots)
    assert len(shots_allocation) == 3
    shots_per_term = total_shots // 3
    for setting in shots_allocation:
        assert setting.n_shots == shots_per_term


def test_equipartition_grouped_with_unit(operator: Operator) -> None:
    groups = {
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    }
    total_shots = 1111
    shot_unit = 10
    allocator = create_equipartition_shots_allocator(shot_unit)
    shots_allocation = allocator(operator, groups, total_shots)
    assert len(shots_allocation) == 3
    shots_per_term = (total_shots // (shot_unit * 3)) * shot_unit
    for setting in shots_allocation:
        assert setting.n_shots == shots_per_term


def test_proportional_individual(operator: Operator) -> None:
    groups = {
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1]}),
        frozenset({PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    }
    total_shots = 1111
    allocator = create_proportional_shots_allocator()
    shots_allocation = allocator(operator, groups, total_shots)
    assert shots_allocation == {
        PauliSamplingSetting(frozenset({PAULI_LABELS[0]}), 44),
        PauliSamplingSetting(frozenset({PAULI_LABELS[1]}), 333),
        PauliSamplingSetting(frozenset({PAULI_LABELS[2]}), 444),
        PauliSamplingSetting(frozenset({PAULI_LABELS[4]}), 288),
    }
    total_shots_actual = sum(setting.n_shots for setting in shots_allocation)
    assert total_shots_actual <= total_shots


def test_proportional_grouped(operator: Operator) -> None:
    groups = {
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    }
    total_shots = 1111
    allocator = create_proportional_shots_allocator()
    shots_allocation = allocator(operator, groups, total_shots)
    assert shots_allocation == {
        PauliSamplingSetting(frozenset({PAULI_LABELS[0]}), 55),
        PauliSamplingSetting(frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}), 694),
        PauliSamplingSetting(frozenset({PAULI_LABELS[4]}), 361),
    }
    total_shots_actual = sum(setting.n_shots for setting in shots_allocation)
    assert total_shots_actual <= total_shots


def test_proportional_grouped_with_unit(operator: Operator) -> None:
    groups = {
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    }
    total_shots = 1111
    shot_unit = 10
    allocator = create_proportional_shots_allocator(shot_unit)
    shots_allocation = allocator(operator, groups, total_shots)
    assert shots_allocation == {
        PauliSamplingSetting(frozenset({PAULI_LABELS[0]}), 50),
        PauliSamplingSetting(frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}), 690),
        PauliSamplingSetting(frozenset({PAULI_LABELS[4]}), 360),
    }
    total_shots_actual = sum(setting.n_shots for setting in shots_allocation)
    assert total_shots_actual <= total_shots


@patch("quri_parts.core.sampling.shots_allocator.default_rng")
def test_weighted_random_individual(default_rng: Mock, operator: Operator) -> None:
    groups = (
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1]}),
        frozenset({PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    )  # use tuple for fixing the order of groups (only for testing)
    seed = 524
    dummy_multinomial_result = [100, 200, 300, 400]
    total_shots = sum(dummy_multinomial_result)

    allocator = create_weighted_random_shots_allocator(seed)
    default_rng.assert_called_with(seed)

    rng = default_rng.return_value
    rng.multinomial.return_value = np.array([dummy_multinomial_result])
    shots_allocation = allocator(operator, groups, total_shots)
    rng.multinomial.assert_called_once()

    weights_expected = [0.04, 0.3, 0.4, 0.26]
    args, kwargs = rng.multinomial.call_args
    assert args == (total_shots, weights_expected)
    assert kwargs == {"size": 1}

    assert shots_allocation == {
        PauliSamplingSetting(groups[0], 100),
        PauliSamplingSetting(groups[1], 200),
        PauliSamplingSetting(groups[2], 300),
        PauliSamplingSetting(groups[3], 400),
    }


@patch("quri_parts.core.sampling.shots_allocator.default_rng")
def test_weighted_random_grouped(default_rng: Mock, operator: Operator) -> None:
    groups = (
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    )  # use tuple for fixing the order of groups (only for testing)
    seed = 524
    dummy_multinomial_result = [100, 200, 300]
    total_shots = sum(dummy_multinomial_result)

    allocator = create_weighted_random_shots_allocator(seed)
    default_rng.assert_called_with(seed)

    rng = default_rng.return_value
    rng.multinomial.return_value = np.array([dummy_multinomial_result])
    shots_allocation = allocator(operator, groups, total_shots)
    rng.multinomial.assert_called_once()

    weights_expected = [0.05, 0.625, 0.325]
    args, kwargs = rng.multinomial.call_args
    assert args == (total_shots, weights_expected)
    assert kwargs == {"size": 1}
    assert shots_allocation == {
        PauliSamplingSetting(groups[0], 100),
        PauliSamplingSetting(groups[1], 200),
        PauliSamplingSetting(groups[2], 300),
    }


@patch("quri_parts.core.sampling.shots_allocator.default_rng")
def test_weighted_random_grouped_with_unit(
    default_rng: Mock, operator: Operator
) -> None:
    groups = (
        frozenset({PAULI_LABELS[0]}),
        frozenset({PAULI_LABELS[1], PAULI_LABELS[2]}),
        frozenset({PAULI_LABELS[4]}),
    )  # use tuple for fixing the order of groups (only for testing)
    seed = 524
    total_shots = 666
    shot_unit = 100
    dummy_multinomial_result = [1, 2, 3]

    allocator = create_weighted_random_shots_allocator(seed, shot_unit)
    default_rng.assert_called_with(seed)

    rng = default_rng.return_value
    rng.multinomial.return_value = np.array([dummy_multinomial_result])
    shots_allocation = allocator(operator, groups, total_shots)
    rng.multinomial.assert_called_once()

    weights_expected = [0.05, 0.625, 0.325]
    args, kwargs = rng.multinomial.call_args
    assert args == (6, weights_expected)  # 6: 666 // 100
    assert kwargs == {"size": 1}
    assert shots_allocation == {
        PauliSamplingSetting(groups[0], 100),
        PauliSamplingSetting(groups[1], 200),
        PauliSamplingSetting(groups[2], 300),
    }
