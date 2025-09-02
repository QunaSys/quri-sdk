# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from collections.abc import Mapping
from typing import Union, cast

import numpy as np

from quri_parts.core.measurement import (
    PauliReconstructorFactory,
    bitwise_pauli_reconstructor_factory,
)
from quri_parts.core.operator import PAULI_IDENTITY, CommutablePauliSet, PauliLabel
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.utils.statistics import count_var


def general_pauli_expectation_estimator(
    counts: MeasurementCounts,
    pauli: PauliLabel,
    reconstructor_factory: PauliReconstructorFactory,
) -> float:
    """An implementation of :class:`~PauliExpectationEstimator` for a given
    :class:`PauliReconstructorFactory`."""
    if len(counts) == 0:
        raise ValueError("No measurement counts supplied (counts is empty).")

    if pauli == PAULI_IDENTITY:
        return 1.0

    reconstructor = reconstructor_factory(pauli)

    val = 0.0
    for key, count in counts.items():
        val += reconstructor(key) * count
    val /= sum(counts.values())
    return val


def general_pauli_sum_expectation_estimator(
    counts: MeasurementCounts,
    pauli_set: CommutablePauliSet,
    coefs: Mapping[PauliLabel, complex],
    reconstructor_factory: PauliReconstructorFactory,
) -> complex:
    """Estimate expectation value of a weighted sum of commutable Pauli
    operators from measurement counts and Pauli reconstructor.

    Note that this function calculates the sum for only Pauli operators
    contained in both of ``pauli_set`` and ``coefs``.
    """
    pauli_exp_and_coefs = [
        (
            general_pauli_expectation_estimator(counts, pauli, reconstructor_factory),
            coefs[pauli],
        )
        for pauli in pauli_set
        if pauli in coefs
    ]
    if not pauli_exp_and_coefs:
        return 0
    pauli_exp_seq, coef_seq = zip(*pauli_exp_and_coefs)
    return cast(complex, np.inner(pauli_exp_seq, coef_seq))


def general_pauli_covariance_estimator(
    counts: MeasurementCounts,
    pauli1: PauliLabel,
    pauli2: PauliLabel,
    reconstructor_factory: PauliReconstructorFactory,
) -> float:
    """An implementation of :class:`~PauliCovarianceEstimator` for a given
    :class:`PauliReconstructorFactory`."""
    if len(counts) == 0:
        raise ValueError("No measurement counts supplied (counts is empty).")

    if pauli1 == PAULI_IDENTITY or pauli2 == PAULI_IDENTITY:
        return 0.0

    if len(counts) == 1:
        # When there is only one count, the variables have fixed values and
        # the covariance is zero.
        return 0.0

    reconstructor1 = reconstructor_factory(pauli1)
    reconstructor2 = reconstructor_factory(pauli2)

    p1_sum = 0.0
    p2_sum = 0.0
    p12_sum = 0.0
    for key, count in counts.items():
        p1 = reconstructor1(key)
        p2 = reconstructor2(key)
        p1_sum += p1 * count
        p2_sum += p2 * count
        p12_sum += p1 * p2 * count
    s = sum(counts.values())
    return (p12_sum - p1_sum * p2_sum / s) / (s - 1)


def general_pauli_sum_sample_variance(
    counts: MeasurementCounts,
    pauli_set: CommutablePauliSet,
    coefs: Mapping[PauliLabel, complex],
    reconstructor_factory: PauliReconstructorFactory,
) -> float:
    """Calculate sample variance of a weighted sum of commutable Pauli
    operators from measurement counts and Pauli reconstructor.

    Note that this function calculates the variance of sum for only
    Pauli operators contained in both of ``pauli_set`` and ``coefs``.
    """
    pauli_values = defaultdict(list)
    for pauli in pauli_set:
        if pauli not in coefs:
            continue
        reconstructor = reconstructor_factory(pauli)
        c = coefs[pauli]
        for bits in counts:
            val = reconstructor(bits) * c
            pauli_values[bits].append(val)
    value_count_list = [
        (sum(pauli_values[bits]), count) for bits, count in counts.items()
    ]
    value_counts: dict[complex, Union[int, float]] = defaultdict(float)
    for val, count in value_count_list:
        value_counts[val] += count
    return count_var(value_counts)


def trivial_pauli_expectation_estimator(
    counts: MeasurementCounts, pauli: PauliLabel
) -> float:
    r"""An implementation of :class:`~PauliExpectationEstimator`, which assumes
    a "trivial" measurement of the Pauli string.

    The "trivial" measurement of a Pauli string refers to a measurement
    in computational basis preceded by single qubit rotations to map
    each Pauli matrix on each qubit to :math:`Z`. For example, if the
    Pauli string contains :math:`X` (:math:`Y`) at a qubit index
    :math:`i`, then (an :math:`S^\dagger` gate and) an Hadamard gate is
    applied on the qubit :math:`i` before performing the :math:`Z`
    measurements on all qubits.
    """
    return general_pauli_expectation_estimator(
        counts, pauli, bitwise_pauli_reconstructor_factory
    )


def trivial_pauli_covariance_estimator(
    counts: MeasurementCounts, pauli1: PauliLabel, pauli2: PauliLabel
) -> float:
    r"""An implementation of :class:`~PauliCovarianceEstimator`, which assumes a
    "trivial" measurement of the Pauli string.

    The "trivial" measurement of a Pauli string refers to a measurement
    in computational basis preceded by single qubit rotations to map
    each Pauli matrix on each qubit to :math:`Z`. For example, if the
    Pauli string contains :math:`X` (:math:`Y`) at a qubit index
    :math:`i`, then (an :math:`S^\dagger` gate and) an Hadamard gate is
    applied on the qubit :math:`i` before performing the :math:`Z`
    measurements on all qubits.
    """
    return general_pauli_covariance_estimator(
        counts, pauli1, pauli2, bitwise_pauli_reconstructor_factory
    )
