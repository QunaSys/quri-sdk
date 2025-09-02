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
from typing import TYPE_CHECKING, Any, Sequence

from quri_algo.algo.phase_estimation.interface import (
    PhaseEstimationBase,
    PhaseEstimationResult,
)
from quri_algo.core.estimator import StateT

from .utils.exp_val_collector import UnitaryOpPowerSamplingEstimator

if TYPE_CHECKING:
    from .utils.signal import SPEDiscreteSignalFunction


@dataclass
class SPEResult(PhaseEstimationResult):
    """Result of statistical phase estimation. In addition to the phase
    estimation, the signals collected during the computation is also recorded
    in it.

    Args:
        phase: The estimated phase
        signal_functions: The signals collected during the SPE computation.
    """

    signal_functions: Sequence["SPEDiscreteSignalFunction"]


class StatisticalPhaseEstimation(PhaseEstimationBase[StateT], ABC):
    r"""Base class for statistical phase estimations.

    Args:
        unitary_power_estimator: Any estimator that estimates the expectation value
            :math:`\langle U^k \rangle`, where :math:`U` is a unitary operator and `k`
            is the power.
    """

    @property
    @abstractmethod
    def unitary_power_estimator(self) -> UnitaryOpPowerSamplingEstimator[StateT]:
        ...

    @abstractmethod
    def __call__(self, state: StateT, *args: Any, **kwds: Any) -> SPEResult:
        ...
