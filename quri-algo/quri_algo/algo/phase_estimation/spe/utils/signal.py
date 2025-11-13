# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import cached_property
from typing import Callable, Generic, Sequence, cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from quri_algo.algo.phase_estimation.spe.utils.fourier.coefficient import (
    SPEFourierCoefficient,
)
from quri_algo.core.estimator import StateT

from .exp_val_collector import (
    ExpectationValueCollector,
    SPESample,
    UnitaryOpPowerSamplingEstimator,
)
from .fourier.gaussian_coefficient import GaussianSampler
from .fourier.step_func_coefficient import StepFunctionSampler

ClassicalSampler: TypeAlias = Callable[[int], Sequence[SPEFourierCoefficient]]


class SPEDiscreteSignalFunction:
    """Represents the signal Z_F(x) which is evaluated by discrete Fourier
    transform.

    Args:
        spe_samples: Sample results from :class:`ExpectationValueCollector`.
        norm_factor: Normalization factor of computing the Fourier integral.
        exponent_factor: The coefficient c in the Fourier transform
            exponent :math:`e^{-i c k x}`.
    """

    def __init__(
        self,
        spe_samples: Sequence[SPESample],
        norm_factor: float,
        exponent_factor: float = 1.0,
    ):
        self.spe_samples = spe_samples
        self.norm_factor = norm_factor
        self.exponent_factor = exponent_factor

    @cached_property
    def ks(self) -> npt.NDArray[np.float64]:
        """The Fourier modes labels."""
        return np.array(
            [sample.classical_sample.k for sample in self.spe_samples], dtype=np.float64
        )

    @cached_property
    def conv_fourier_coeff(self) -> npt.NDArray[np.complex128]:
        """The fourier coefficients of F(x).

        The convention here is that index 0 of the array corresponds to
        the smallest value of k.
        """
        return np.array(
            [sample.classical_sample.get_coeff() for sample in self.spe_samples],
            dtype=np.complex128,
        )

    @cached_property
    def density_func_coeff(self) -> npt.NDArray[np.complex128]:
        """The fourier coefficient of the spectral density function."""
        return np.array(
            [sample.op_exp_val for sample in self.spe_samples], dtype=np.complex128
        )

    @overload
    def __call__(self, x: float, derivative_order: int = 0) -> complex:
        ...

    @overload
    def __call__(
        self, x: npt.NDArray[np.float64], derivative_order: int = 0
    ) -> npt.NDArray[np.complex128]:
        ...

    def __call__(
        self, x: float | npt.NDArray[np.float64], derivative_order: int = 0
    ) -> complex | npt.NDArray[np.complex128]:
        exponent = np.exp(1j * self.exponent_factor * np.outer(self.ks, x))
        conv_coeff = self.conv_fourier_coeff * self.density_func_coeff
        conv_coeff = conv_coeff * (
            (1j * self.exponent_factor * self.ks) ** derivative_order
        )
        return cast(
            complex | npt.NDArray[np.complex128],
            self.norm_factor * (conv_coeff @ exponent),
        )


class SPEDiscreteSignalGenerator(Generic[StateT]):
    """Base object that generates a :class:`SignalFunction` in a Monte-Carlo
    manner."""

    def __init__(
        self,
        power_estimator: UnitaryOpPowerSamplingEstimator[StateT],
        state: StateT,
        classical_sampler: ClassicalSampler,
        norm_factor: float,
        exponent_factor: float = 1.0,
    ):
        self.power_estimator = power_estimator
        self.state = state
        self.classical_sampler = classical_sampler
        self.norm_factor = norm_factor
        self.exponent_factor = exponent_factor
        # _exp_val_collector: the object that generates the list of <e^{-ik_n x}>.
        self._exp_val_collector = ExpectationValueCollector(
            self.power_estimator, self.state
        )

    def __call__(self, n_sample: int) -> SPEDiscreteSignalFunction:
        r"""Generates the signal function:

            .. math::
                Z_F(x) = \sum_{n} \tilde{F}(k_n) \langle e^{-ik_n x} \rangle

        Args:
            n_sample: number of Monte-Carlo samples.
        """
        # classical_samples: list of Fourier coefficient: \tilde{F}(k_n).
        # spe_samples: list of \\tilde{F}(k_n) <e^{-ik_n x}>

        classical_samples = self.classical_sampler(n_sample)
        spe_samples = self._exp_val_collector(classical_samples)
        return SPEDiscreteSignalFunction(
            spe_samples, self.norm_factor, self.exponent_factor
        )


class StepFunctionSignalGenerator(SPEDiscreteSignalGenerator[StateT]):
    """A factory that generates Z_F(x) based on SPE, where F is the periodic
    step function."""

    def __init__(
        self,
        power_estimator: UnitaryOpPowerSamplingEstimator[StateT],
        state: StateT,
        d: int,
        delta: float,
    ):
        self.d = d
        self.delta = delta
        self.classical_sampler = StepFunctionSampler(self.d, self.delta)
        self.norm_factor = cast(
            float, np.linalg.norm(self.classical_sampler.fourier_coefficients, 1)
        )
        super().__init__(
            power_estimator,
            state,
            self.classical_sampler,
            self.norm_factor,
            exponent_factor=1.0,
        )


class GaussianSignalGenerator(SPEDiscreteSignalGenerator[StateT]):
    r"""A factory that generates Z_F(x) based on SPE, where F is the Gaussian
    distribition function."""

    def __init__(
        self,
        power_estimator: UnitaryOpPowerSamplingEstimator[StateT],
        state: StateT,
        cutoff_T: float,
        n_discretize: int,
        sigma: float,
    ):
        self.cutoff_T = cutoff_T
        self.n_discretize = n_discretize
        self.sigma = sigma
        self.classical_sampler = GaussianSampler(
            self.cutoff_T, self.n_discretize, self.sigma
        )
        self.norm_factor = cast(
            float, np.linalg.norm(self.classical_sampler.fourier_coefficients, 1)
        )
        self.norm_factor *= 2 * self.cutoff_T / self.n_discretize
        super().__init__(
            power_estimator,
            state,
            self.classical_sampler,
            self.norm_factor,
            exponent_factor=2 * np.pi,
        )
