# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

from typing_extensions import TypeAlias

from quri_parts.core.operator import PauliLabel
from quri_parts.core.sampling import MeasurementCounts

from .estimator import (
    concurrent_sampling_estimate,
    create_sampling_concurrent_estimator,
    create_sampling_estimator,
    sampling_estimate,
)
from .overlap_estimator import (
    create_sampling_overlap_estimator,
    create_sampling_overlap_weighted_sum_estimator,
    sampling_overlap_estimate,
    sampling_overlap_weighted_sum_estimate,
)
from .pauli import (
    general_pauli_covariance_estimator,
    general_pauli_expectation_estimator,
    general_pauli_sum_expectation_estimator,
    general_pauli_sum_sample_variance,
    trivial_pauli_covariance_estimator,
    trivial_pauli_expectation_estimator,
)

#: PauliExpectationEstimator represents a function that receives a MeasurementCounts
#: obtained from a sampling measurement and a PauliLabel, and returns an estimate of
#: the expectation value (sample mean) of the pauli string.
PauliExpectationEstimator: TypeAlias = Callable[
    [MeasurementCounts, PauliLabel],
    float,
]

#: PauliCovarianceEstimator represents a function that receives a MeasurementCounts
#: obtained from a sampling measurement and two PauliLabel's, and returns an estimate
#: of the covariance (sample covariance) of the two pauli strings. It is assumed that
#: the two PauliLabel's commute.
PauliCovarianceEstimator: TypeAlias = Callable[
    [MeasurementCounts, PauliLabel, PauliLabel], float
]

__all__ = [
    "PauliExpectationEstimator",
    "PauliCovarianceEstimator",
    "general_pauli_expectation_estimator",
    "general_pauli_sum_expectation_estimator",
    "general_pauli_covariance_estimator",
    "general_pauli_sum_sample_variance",
    "trivial_pauli_expectation_estimator",
    "trivial_pauli_covariance_estimator",
    "sampling_estimate",
    "create_sampling_estimator",
    "concurrent_sampling_estimate",
    "create_sampling_concurrent_estimator",
    "sampling_overlap_estimate",
    "create_sampling_overlap_estimator",
    "sampling_overlap_weighted_sum_estimate",
    "create_sampling_overlap_weighted_sum_estimator",
]
