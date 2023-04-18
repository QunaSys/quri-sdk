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
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, cast

from qulacs import QuantumState
from qulacs.state import inner_product

from quri_parts.core.estimator import (
    Estimate,
    OverlapEstimator,
    OverlapWeightedSumEstimator,
    ParametricOverlapWeightedSumEstimator,
)
from quri_parts.core.state import ParametricQuantumStateVector, QuantumStateVector
from quri_parts.core.utils.concurrent import execute_concurrently
from quri_parts.qulacs import QulacsParametricStateT, QulacsStateT

from . import cast_to_list

if TYPE_CHECKING:
    from concurrent.futures import Executor

from .circuit import convert_circuit


class _WeightedSumEstimate(NamedTuple):
    value: complex
    error: float = 0.0


class _Estimate(NamedTuple):
    value: float
    error: float = 0.0


def _create_qulacs_initial_state(
    state: Union[QulacsStateT, QulacsParametricStateT]
) -> QuantumState:
    qs_state = QuantumState(state.qubit_count)
    if isinstance(state, (QuantumStateVector, ParametricQuantumStateVector)):
        qs_state.load(cast_to_list(state.vector))
    return qs_state


def _estimate(ket: QulacsStateT, bra: QulacsStateT) -> Estimate[float]:
    ket_circuit = convert_circuit(ket.circuit)
    bra_circuit = convert_circuit(bra.circuit)
    qulacs_ket = _create_qulacs_initial_state(ket)
    qulacs_bra = _create_qulacs_initial_state(bra)
    ket_circuit.update_quantum_state(qulacs_ket)
    bra_circuit.update_quantum_state(qulacs_bra)
    overlap = inner_product(qulacs_ket, qulacs_bra)
    overlap_mag_sqrd = abs(overlap) ** 2

    return _Estimate(value=overlap_mag_sqrd)


def create_qulacs_vector_overlap_estimator() -> OverlapEstimator[QulacsStateT]:
    """Returns an :class:`~OverlapEstimator` that uses Qulacs vector simulator
    to calculate magnitude squared overlaps."""

    return _estimate


def _weighted_sum_estimate(
    kets: Sequence[QulacsStateT],
    bras: Sequence[QulacsStateT],
    weights: Sequence[complex],
    executor: Optional["Executor"],
    concurrency: int = 1,
) -> Estimate[complex]:
    len_kets = len(kets)
    len_bras = len(bras)
    len_weights = len(weights)

    if any(len_kets != x for x in {len_bras, len_weights}):
        raise ValueError(
            f"Number of kets ({len_kets}) and "
            f"number of bras ({len_bras}) and "
            f"number of weights ({len_weights}) does not match."
        )

    def _sequential_estimate(
        _: Any,
        input: Sequence[tuple[QulacsStateT, QulacsStateT]],
    ) -> Sequence[Estimate[complex]]:
        return [_estimate(ket, bra) for ket, bra in input]

    overlap_estimates = execute_concurrently(
        _sequential_estimate,
        None,
        zip(kets, bras),
        executor,
        concurrency,
    )

    return _WeightedSumEstimate(
        sum(
            weight * estimate.value
            for estimate, weight in zip(overlap_estimates, weights)
        )
    )


def create_qulacs_vector_overlap_weighted_sum_estimator(
    executor: Optional["Executor"], concurrency: int = 1
) -> OverlapWeightedSumEstimator[QulacsStateT]:  # noqa: E501
    """Returns an :class:`~OverlapEstimator` that uses Qulacs vector simulator
    to calculate magnitude squared overlaps."""

    def estimator(
        kets: Sequence[QulacsStateT],
        bras: Sequence[QulacsStateT],
        weights: Sequence[complex],
    ) -> Estimate[complex]:
        return _weighted_sum_estimate(kets, bras, weights, executor, concurrency)

    return estimator


def create_qulacs_vector_parametric_overlap_weighted_sum_estimator(
    estimator: OverlapWeightedSumEstimator[QulacsStateT],
) -> ParametricOverlapWeightedSumEstimator[QulacsParametricStateT]:
    """Create a :class:`ParametricOverlapWeightedSumEstimator` from an
    :class:`ParametricOverlapWeightedSumEstimator` that estimates
    weighted magnitude squared overlap values by qulacs statevector evaluation.
    """

    def parametric_estimator(
        kets: tuple[QulacsParametricStateT, Sequence[Sequence[float]]],
        bras: tuple[QulacsParametricStateT, Sequence[Sequence[float]]],
        weights: Sequence[complex],
    ) -> Estimate[complex]:
        bound_kets = [
            cast(QulacsStateT, kets[0].bind_parameters(params)) for params in kets[1]
        ]
        bound_bras = [
            cast(QulacsStateT, bras[0].bind_parameters(params)) for params in bras[1]
        ]

        return estimator(bound_kets, bound_bras, weights)

    return parametric_estimator
