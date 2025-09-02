# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.core.estimator.sampling import (
    general_pauli_sum_expectation_estimator,
    general_pauli_sum_sample_variance,
    trivial_pauli_covariance_estimator,
    trivial_pauli_expectation_estimator,
)
from quri_parts.core.measurement import bitwise_pauli_reconstructor_factory
from quri_parts.core.operator import PAULI_IDENTITY, pauli_label


class TestGeneralPauliSumExpectationEstimator:
    def test_raise_when_counts_empty(self) -> None:
        with pytest.raises(ValueError):
            general_pauli_sum_expectation_estimator(
                {},
                {PAULI_IDENTITY},
                {PAULI_IDENTITY: 1.0},
                bitwise_pauli_reconstructor_factory,
            )

    def test_estimate_correctly(self) -> None:
        op = {
            pauli_label("X2 Y4 Z6"): 2.0 + 3.0j,
            pauli_label("X1 Y3 Z5"): 0.5 + 2.0j,
            PAULI_IDENTITY: 1.5 + 1.0j,
        }
        pauli_set = {*op.keys()}

        counts = {
            0: 16,  # no bit set = (1, 1)
            0b100: 32,  # bit set on X2 = (-1, 1)
            0b11100: 128,  # bit set on X2, Y3, Y4 = (1, -1)
            0b1001100: 16,  # bit set on X2, Y3, Z6 = (1, -1)
            0b1001010101: 64,  # bit set on X2, Y4, Z6 = (-1, 1)
        }
        total = sum(counts.values())

        exp = general_pauli_sum_expectation_estimator(
            counts, pauli_set, op, bitwise_pauli_reconstructor_factory
        )
        expected = (
            (16 - 32 + 128 + 16 - 64) * (2.0 + 3.0j)
            + (16 + 32 - 128 - 16 + 64) * (0.5 + 2.0j)
        ) / total + (1.5 + 1.0j)
        assert exp == expected

    def test_estimate_correctly_for_larger_pauli_set(self) -> None:
        op = {
            pauli_label("X2 Y4 Z6"): 2.0 + 3.0j,
            pauli_label("X1 Y3 Z5"): 0.5 + 2.0j,
            PAULI_IDENTITY: 1.5 + 1.0j,
        }
        pauli_set = {pauli_label("X2 Y4 Z6"), pauli_label("X1 Z3 Y5")}

        counts = {
            0: 16,  # no bit set = 1
            0b100: 32,  # bit set on X2 = -1
            0b11100: 128,  # bit set on X2, Y4 = 1
            0b1001100: 16,  # bit set on X2, Z6 = 1
            0b1001010101: 64,  # bit set on X2, Y4, Z6 = -1
        }
        total = sum(counts.values())

        exp = general_pauli_sum_expectation_estimator(
            counts, pauli_set, op, bitwise_pauli_reconstructor_factory
        )
        expected = ((16 - 32 + 128 + 16 - 64) * (2.0 + 3.0j)) / total
        assert exp == expected


class TestGeneralPauliSumSampleVariance:
    def test_raise_when_counts_empty(self) -> None:
        with pytest.raises(ValueError):
            general_pauli_sum_sample_variance(
                {},
                {PAULI_IDENTITY},
                {PAULI_IDENTITY: 1.0},
                bitwise_pauli_reconstructor_factory,
            )

    def test_estimate_correctly(self) -> None:
        c1 = 2.0 + 3.0j
        c2 = 0.5 + 2.0j
        c0 = 1.5 + 1.0j
        op = {
            pauli_label("X2 Y4 Z6"): c1,
            pauli_label("X1 Y3 Z5"): c2,
            PAULI_IDENTITY: c0,
        }
        pauli_set = {*op.keys()}

        counts = {
            0: 16,  # no bit set = (1, 1)
            0b100: 32,  # bit set on X2 = (-1, 1)
            0b11100: 128,  # bit set on X2, Y3, Y4 = (1, -1)
            0b1001100: 16,  # bit set on X2, Y3, Z6 = (1, -1)
            0b1001010101: 64,  # bit set on X2, Y4, Z6 = (-1, 1)
        }
        total = sum(counts.values())

        mean = ((16 - 32 + 128 + 16 - 64) * c1 + (16 + 32 - 128 - 16 + 64) * c2) / total

        sv = general_pauli_sum_sample_variance(
            counts, pauli_set, op, bitwise_pauli_reconstructor_factory
        )
        expected = (
            16 * abs(c1 + c2 - mean) ** 2
            + 32 * abs(-c1 + c2 - mean) ** 2
            + 128 * abs(c1 - c2 - mean) ** 2
            + 16 * abs(c1 - c2 - mean) ** 2
            + 64 * abs(-c1 + c2 - mean) ** 2
        ) / total
        assert sv == expected

    def test_estimate_correctly_for_larger_pauli_set(self) -> None:
        c1 = 2.0 + 3.0j
        c2 = 0.5 + 2.0j
        c0 = 1.5 + 1.0j
        op = {
            pauli_label("X2 Y4 Z6"): c1,
            pauli_label("X1 Y3 Z5"): c2,
            PAULI_IDENTITY: c0,
        }
        pauli_set = {pauli_label("X2 Y4 Z6"), pauli_label("X1 Z3 Y5")}

        counts = {
            0: 16,  # no bit set = 1
            0b100: 32,  # bit set on X2 = -1
            0b11100: 128,  # bit set on X2, Y4 = 1
            0b1001100: 16,  # bit set on X2, Z6 = 1
            0b1001010101: 64,  # bit set on X2, Y4, Z6 = -1
        }
        total = sum(counts.values())

        mean = (16 - 32 + 128 + 16 - 64) * c1 / total

        sv = general_pauli_sum_sample_variance(
            counts, pauli_set, op, bitwise_pauli_reconstructor_factory
        )
        expected = (
            16 * abs(c1 - mean) ** 2
            + 32 * abs(-c1 - mean) ** 2
            + 128 * abs(c1 - mean) ** 2
            + 16 * abs(c1 - mean) ** 2
            + 64 * abs(-c1 - mean) ** 2
        ) / total
        assert sv == expected


class TestTrivialPauliExpectationEstimator:
    def test_raise_when_counts_empty(self) -> None:
        with pytest.raises(ValueError):
            trivial_pauli_expectation_estimator({}, PAULI_IDENTITY)

    def test_always_return_one_for_identity(self) -> None:
        exp = trivial_pauli_expectation_estimator({0: 16, 1: 16}, PAULI_IDENTITY)
        assert exp == 1.0

    def test_estimate_correctly_from_single_count(self) -> None:
        pauli = pauli_label("X2 Y4 Z6")

        # no bit set
        exp = trivial_pauli_expectation_estimator({0: 16}, pauli)
        assert exp == 1.0

        # bit set on X2
        exp = trivial_pauli_expectation_estimator({0b100: 16}, pauli)
        assert exp == -1.0

        # bit set on X2, Y4
        exp = trivial_pauli_expectation_estimator({0b11100: 16}, pauli)
        assert exp == 1.0

        # bit set on X2, Z6
        exp = trivial_pauli_expectation_estimator({0b1001100: 16}, pauli)
        assert exp == 1.0

        # bit set on X2, Y4, Z6
        exp = trivial_pauli_expectation_estimator({0b1001010101: 16}, pauli)
        assert exp == -1.0

    def test_estimate_correctly_from_multiple_counts(self) -> None:
        pauli = pauli_label("X2 Y4 Z6")

        counts = {
            0: 16,  # no bit set = 1
            0b100: 32,  # bit set on X2 = -1
            0b11100: 128,  # bit set on X2, Y4 = 1
            0b1001100: 16,  # bit set on X2, Z6 = 1
            0b1001010101: 64,  # bit set on X2, Y4, Z6 = -1
        }
        total = sum(counts.values())

        exp = trivial_pauli_expectation_estimator(counts, pauli)
        assert exp == (16 - 32 + 128 + 16 - 64) / total


class TestTrivialPauliCovarianceEstimator:
    def test_raise_when_counts_empty(self) -> None:
        with pytest.raises(ValueError):
            trivial_pauli_covariance_estimator({}, PAULI_IDENTITY, PAULI_IDENTITY)

    def test_always_return_zero_for_identity(self) -> None:
        pauli = pauli_label("Z0")

        cov = trivial_pauli_covariance_estimator(
            {0: 16, 1: 16}, PAULI_IDENTITY, PAULI_IDENTITY
        )
        assert cov == 0.0

        cov = trivial_pauli_covariance_estimator({0: 16, 1: 16}, pauli, PAULI_IDENTITY)
        assert cov == 0.0

        cov = trivial_pauli_covariance_estimator({0: 16, 1: 16}, PAULI_IDENTITY, pauli)
        assert cov == 0.0

    def test_always_return_zero_when_only_one_count_available(self) -> None:
        pauli1 = pauli_label("X2 Y4 Z6")
        pauli2 = pauli_label("X3 Z6")

        for b in [0, 0b100, 0b11100, 0b1001100, 0b1001010101]:
            cov = trivial_pauli_covariance_estimator({b: 16}, pauli1, pauli2)
            assert cov == 0.0

    def test_estimate_correctly_from_multiple_counts(self) -> None:
        pauli1 = pauli_label("X2 Y4 Z6")
        pauli2 = pauli_label("X3 Z6")

        counts = {
            0: 16,  # no bit set: product = 1
            0b100: 32,  # bit set on X2: product = -1
            0b1000: 64,  # bit set on X3: product = -1
            0b1100: 16,  # bit set on X2, X3: product = 1
            0b11100: 128,  # bit set on X2, X3, X4: product = -1
            0b1001100: 16,  # bit set on X2, X3, Z6: product = 1
            0b1001010101: 64,  # bit set on X2, Y4, Z6: product = 1
        }
        total = sum(counts.values())

        exp1 = trivial_pauli_expectation_estimator(counts, pauli1)
        exp2 = trivial_pauli_expectation_estimator(counts, pauli2)
        cov = trivial_pauli_covariance_estimator(counts, pauli1, pauli2)
        expected = ((16 - 32 - 64 + 16 - 128 + 16 + 64) - exp1 * exp2 * total) / (
            total - 1
        )
        assert cov == pytest.approx(expected, rel=1 / total)
