# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Generic, NamedTuple, Optional, Sequence

from quri_parts.core.estimator import Estimate
from typing_extensions import TypeAlias

from quri_algo.core.estimator import StateT

from .fourier.coefficient import SPEFourierCoefficient

UnitaryOpPowerSamplingEstimator: TypeAlias = Callable[
    [StateT, int | float, Optional[int]], Estimate[complex]
]


class SPESample(NamedTuple):
    r"""Holds the classically sampled Fourier coefficient
    :math:`\widetilde{F}(k)` along with the Fourier coefficient of the spectral
    density function :math:`\widetilde{\varrho}(k)`

    Args:
        classical_sample: The Fourier coefficient :math:`\widetilde{F}(k)`.
        op_exp_val:
            The spectral density function's Fourier coefficient
            :math:`\widetilde{\varrho}`(k). This should be the same as
            :\langle U^k \rangle:
    """

    classical_sample: SPEFourierCoefficient
    op_exp_val: complex


class ExpectationValueCollector(Generic[StateT]):
    r"""Computes the expectation value :math:`\langle \psi|U^{k}|\psi\rangle`
    for a given sequence of :class:`SPEFourierCoefficient`. This sets up the
    convolution function's Fourier coefficient.

    ..math::
        \langle \psi|U^{k}|\psi\rangle \widetilde{F}(k)
    """

    def __init__(
        self,
        operator_power_estimator: UnitaryOpPowerSamplingEstimator[StateT],
        state: StateT,
    ):
        self.operator_power_estimator = operator_power_estimator
        self.state = state

    def __call__(
        self, classical_samples: Sequence[SPEFourierCoefficient]
    ) -> Sequence[SPESample]:
        """
        classical_samples: list of :math:`\\tilde{F}(k)`
        """
        sample: list[SPESample] = []
        for c_sample in classical_samples:
            exp_val = self.operator_power_estimator(
                self.state, c_sample.k, c_sample.n_distribute
            )
            spe_sample = SPESample(c_sample, exp_val.value)
            sample.append(spe_sample)
        return sample
