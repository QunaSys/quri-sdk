# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Sequence

import numpy as np
from quri_parts.core.estimator import Estimate, Estimates

from quri_algo.core.estimator import StateT


@dataclass
class PhaseEstimationResult(Estimate[float]):
    phase: float

    @property
    def value(self) -> float:
        return self.phase

    @property
    def error(self) -> float:
        return np.nan


@dataclass
class MultiplePhaseEstimationResult(Estimates[float]):
    phases: Sequence[float]

    @property
    def values(self) -> Sequence[float]:
        return self.phases

    @property
    def error_matrix(self) -> Optional[Sequence[Sequence[float]]]:
        return None


class PhaseEstimationBase(Generic[StateT], ABC):
    """Base class for implementing phase estimation algorithm."""

    @abstractmethod
    def __call__(
        self, state: StateT, *args: Any, **kwds: Any
    ) -> PhaseEstimationResult | MultiplePhaseEstimationResult:
        ...
