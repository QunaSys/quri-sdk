# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy import optimize

from quri_algo.core.estimator import StateT
from quri_algo.core.estimator.time_evolution import (
    TimeEvolutionExpectationValueEstimator,
    TimeEvolutionPowerEstimator,
)

from ..interface import SPEResult, StatisticalPhaseEstimation
from ..utils.exp_val_collector import UnitaryOpPowerSamplingEstimator
from ..utils.signal import GaussianSignalGenerator, SPEDiscreteSignalFunction

if TYPE_CHECKING:
    from quri_algo.algo.phase_estimation.spe import GaussianParam


class GaussianFitter:
    """Object that performs least square fitting of a Gaussian distribution
    function."""

    def __init__(
        self,
        sigma: float,
        search_range: npt.NDArray[np.float64],
        estimated_conv_value: npt.NDArray[np.float64],
        bounds: tuple[tuple[float, float], tuple[float, float]] = (
            (-np.pi / 3, np.pi / 3),
            (0, 1),
        ),
    ):
        self.sigma = sigma
        self.search_range = search_range
        self.estimated_conv_value = estimated_conv_value
        self.bounds = bounds

    def get_loss_function(self) -> Callable[[npt.NDArray[np.float64]], float]:
        def f(mu_and_p: npt.NDArray[np.float64]) -> float:
            mu, p = mu_and_p
            xs = self.search_range
            ys = self.estimated_conv_value
            yi = p * np.exp(-((xs - mu) ** 2) / 2 / (self.sigma) ** 2)
            yi /= np.sqrt(2 * np.pi)
            yi /= self.sigma
            return float(np.linalg.norm(ys - yi) ** 2 / len(xs))

        return f

    def fit(self, mu: float, p0: float) -> optimize.OptimizeResult:
        """Performs the least mean square fitting for the target function."""
        loss = self.get_loss_function()
        return optimize.minimize(loss, [mu, p0], bounds=self.bounds)


class GaussianFittingPhaseEstimation(StatisticalPhaseEstimation[StateT]):
    """The Gaussian Fitting statistical phase estimation algorithm.

    Ref:
        The convolution function follows the following reference:
        Guoming Wang, Daniel Stilck França, Ruizhe Zhang, Shuchen Zhu, Peter D. Johnson
        Quantum algorithm for ground state energy estimation using circuit depth with
        exponentially improved dependence on precision
        Quantum 7, 1167 (2023).

        The post-processing of the signal follows the following reference:
        Ming-Zhi Chung, Andreas Thomasen, Henry Liao, Ryosuke Imai
        Contrasting Statistical Phase Estimation with the Variational Quantum
        Eigensolver in the era of Early Fault Tolerant Quantum Computation
        arXiv:2409.07749
    """

    def __init__(
        self, unitary_power_estimator: UnitaryOpPowerSamplingEstimator[StateT]
    ):
        self._unitary_power_estimator = unitary_power_estimator

    @property
    def unitary_power_estimator(self) -> UnitaryOpPowerSamplingEstimator[StateT]:
        return self._unitary_power_estimator

    def __call__(
        self,
        state: StateT,
        gaussian_param: "GaussianParam",
        search_range: npt.NDArray[np.float64],
        eta: float,
        *,
        signal_function: Optional[SPEDiscreteSignalFunction] = None,
    ) -> SPEResult:
        """
        Args:
            state: The input quantum state to run the phase estimation with.
            gaussian_param: The :class:`GaussianParam` to construct the Gaussian
                convolution function.
            search_range: The interval the desired peak falls into.
            eta: Guess of the height of the peak. Usually corresponds to the lower bound
                of the input state's overlap with the desired eigenstate.
            signal_function: A signal function to be post processed with. When this is
                not None, the state will be ignored, i.e. it is equivalent to running:

                >>> GaussianFittingPhaseEstimation.fit_signal(
                ...     signal_function, gaussian_param.sigma, search_range, eta
                ... )
        """
        if signal_function is None:
            signal_generator = GaussianSignalGenerator(
                self.unitary_power_estimator,
                state,
                gaussian_param.T,
                gaussian_param.N,
                gaussian_param.sigma,
            )
            signal_function = signal_generator(gaussian_param.n_sample)
        phase = self.fit_signal(
            signal_function, gaussian_param.sigma, search_range, eta
        )
        return SPEResult(phase=phase, signal_functions=[signal_function])

    @staticmethod
    def fit_signal(
        signal_function: SPEDiscreteSignalFunction,
        sigma: float,
        search_range: npt.NDArray[np.float64],
        eta: float,
    ) -> float:
        signal_pts = signal_function(search_range)
        fitter = GaussianFitter(
            sigma,
            search_range,
            signal_pts.real,
            ((search_range[0], search_range[-1]), (0, 1)),
        )
        guess = search_range[np.argmax(np.abs(signal_pts))]
        opt_result = fitter.fit(guess, eta)
        return float(opt_result.x[0])

    @staticmethod
    def get_recommended_search_range(
        guess_val: float, n_scanned_energy: int, sigma: float, n_sigma: int = 1
    ) -> npt.NDArray[np.float64]:
        M = n_scanned_energy
        sigma = sigma
        return (
            guess_val
            - 0.25 * n_sigma * sigma
            + 0.5 * n_sigma * sigma / M * np.arange(M)
        )


class GaussianFittingGSEE(GaussianFittingPhaseEstimation[StateT]):
    """Performs ground state energy estimation with the Gaussian Fitting
    SPE."""

    def __init__(
        self,
        time_evo_estimator: TimeEvolutionExpectationValueEstimator[StateT],
        tau: float,
    ):
        unitary_power_estimator = TimeEvolutionPowerEstimator(
            time_evo_estimator, 2 * np.pi * tau
        )
        super().__init__(unitary_power_estimator)
