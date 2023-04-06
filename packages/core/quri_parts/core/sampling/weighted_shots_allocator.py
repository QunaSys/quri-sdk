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
from collections.abc import Sequence
from typing import Union

from numpy.random import default_rng

from quri_parts.core.sampling import WeightedSamplingShotsAllocator


def _rounddown_to_unit(n: float, shot_unit: int) -> int:
    return shot_unit * math.floor(n / shot_unit)


def _calc_ratios(weights: Sequence[complex]) -> list[float]:
    weights_sum = sum(map(lambda w: abs(w), weights))
    return list(map(lambda w: abs(w) / weights_sum, weights))


def create_equipartition_generic_shots_allocator(
    shot_unit: int = 1,
) -> WeightedSamplingShotsAllocator:
    """Returns a :class:`~PauliSamplingShotsAllocator` that distributes the
    number of shots equally among the groups.

    Args:
        shot_unit: Unit of shot counts. Each distributed number of shots is
            rounded down to the nearest multiple of this number. (default: 1)

    Note:
        The sum of allocated shots may be less than `total_shots` due to the rounding.
    """

    def allocator(
        weights: Sequence[complex],
        total_shots: int,
    ) -> Sequence[int]:
        n_terms = len(weights)
        shots_per_term = _rounddown_to_unit(total_shots / n_terms, shot_unit)
        return [shots_per_term for _ in range(n_terms)]

    return allocator


def create_proportional_generic_shots_allocator(
    shot_unit: int = 1,
) -> WeightedSamplingShotsAllocator:
    r"""Returns a :class:`~PauliSamplingShotsAllocator` that distributes the
    number of shots proportionately to a set of weights.

    Args:
        shot_unit: Unit of shot counts. Each distributed number of shots is
            rounded down to the nearest multiple of this number. (default: 1)

    Note:
        The sum of allocated shots may be less than `total_shots` due to the rounding.
    """

    def allocator(
        weights: Sequence[complex],
        total_shots: int,
    ) -> Sequence[int]:
        ratios = _calc_ratios(weights)
        shots_list = [
            _rounddown_to_unit(total_shots * ratio, shot_unit) for ratio in ratios
        ]
        return shots_list

    return allocator


def create_weighted_random_generic_shots_allocator(
    seed: int = 1,
    shot_unit: int = 1,
) -> WeightedSamplingShotsAllocator:
    r"""Returns a :class:`~PauliSamplingShotsAllocator` that distributes the
    number of shots according to a multinomial probability distribution defined
    by the target coefficients. The probability is determined by eq. (10) in
    the reference below. Although here we substitute an arbitrary sequence of
    weights for the coefficients mentioned in the paper.

    References:
        Operator Sampling for Shot-frugal Optimization in Variational Algorithms
        Andrew Arrasmith, Lukasz Cincio, Rolando D. Somma, and Patrick J. Coles,
        arXiv:2004.06252 (2020)

    Args:
        seed: Seed used to initialize NumPy's default_rng.
        shot_unit: Unit of shot counts. Each distributed number of shots is
            multiple of this number. (default: 1)

    Note:
        If `shot_unit` is greater than 1, the quotient `total_shots // shot_unit` is
        once distributed to each group by multinomial distribution, and then each of
        them is multiplied by `shot_unit`.
        In this case, the sum of allocated shots may be less than `total_shots`.
    """
    rng = default_rng(seed)

    def allocator(
        weights: Sequence[Union[float, complex]],
        total_shots: int,
    ) -> Sequence[int]:
        ratios = _calc_ratios(weights)
        total_shots_per_shot_unit = total_shots // shot_unit
        shots_per_shot_unit = rng.multinomial(
            total_shots_per_shot_unit, ratios, size=1
        )[0]
        shots_list: Sequence[int] = (shot_unit * shots_per_shot_unit).tolist()
        return shots_list

    return allocator
