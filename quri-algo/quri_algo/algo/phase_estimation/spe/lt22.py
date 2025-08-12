# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import Logger
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

from quri_algo.core.estimator import StateT
from quri_algo.core.estimator.time_evolution import (
    TimeEvolutionExpectationValueEstimator,
    TimeEvolutionPowerEstimator,
)

from .interface import SPEResult, StatisticalPhaseEstimation
from .utils.exp_val_collector import UnitaryOpPowerSamplingEstimator
from .utils.signal import SPEDiscreteSignalFunction, StepFunctionSignalGenerator

if TYPE_CHECKING:
    from quri_algo.algo.phase_estimation.spe import StepFunctionParam


REF_OVERLAP_RATIO = 3 / 4
CONVERGENCE_TOL = 1e-6
MIN_WIDTH = 1e-3


class LT22PostProcessParam(NamedTuple):
    eta: float
    n_batch: int


class LT22PhaseEstimation(StatisticalPhaseEstimation[StateT]):
    r"""The LT22 statistical phase estimation algorithm.

    Ref:
        Lin Lin and Yu Tong,
        Heisenberg-Limited Ground-State Energy Estimation for Early Fault-Tolerant
        Quantum Computers,
        PRX Quantum 3, 010318

    Args:
        unitary_power_estimator: Any estimator that estimates the expectation value
            :math:`\langle U^k \rangle`, where :math:`U` is a unitary operator and `k`
            is the power.
    """

    def __init__(
        self, unitary_power_estimator: UnitaryOpPowerSamplingEstimator[StateT]
    ):
        self._unitary_power_estimator = unitary_power_estimator
        self._logger = Logger(__name__)

    @property
    def unitary_power_estimator(self) -> UnitaryOpPowerSamplingEstimator[StateT]:
        return self._unitary_power_estimator

    def __call__(
        self,
        state: StateT,
        step_function_param: "StepFunctionParam",
        post_processing_param: LT22PostProcessParam,
    ) -> SPEResult:
        """Invert CDF algorithm of PRX Quantum 3, 010318."""
        function_generator = StepFunctionSignalGenerator(
            self.unitary_power_estimator,
            state,
            step_function_param.d,
            step_function_param.delta,
        )
        signal_functions: list[SPEDiscreteSignalFunction] = []
        x0, x1 = -np.pi / 3, np.pi / 3
        cnt = 0
        while x1 - x0 > MIN_WIDTH:
            cnt += 1
            self._logger.info("----------------------------------------------")
            self._logger.info(f"Iteration: {cnt}")
            self._logger.info(f"Searching between [{x0},  {x1}]")

            curr_x = (x0 + x1) / 2

            self._logger.info(f"Current x [{curr_x}]")

            if self.certify(
                function_generator,
                step_function_param.n_sample,
                curr_x,
                post_processing_param,
                signal_functions,
            ):
                x0 = curr_x - 2 * post_processing_param.eta / 3
            else:
                x1 = curr_x + 2 * post_processing_param.eta / 3
            next_x = (x0 + x1) / 2
            if abs(curr_x - next_x) < CONVERGENCE_TOL:
                break
        return SPEResult(phase=next_x, signal_functions=signal_functions)

    def certify(
        self,
        function_generator: StepFunctionSignalGenerator[StateT],
        n_sample: int,
        x: float,
        post_processing_param: LT22PostProcessParam,
        signal_function_list: list[SPEDiscreteSignalFunction],
    ) -> bool:
        eta, n_batch = post_processing_param
        b, c = 0, 0
        cutoff = REF_OVERLAP_RATIO * eta
        self._logger.info(f"cutoff: {cutoff}")

        for i in range(n_batch):
            self._logger.info(f"running batch {i}")
            cdf = function_generator(n_sample)
            signal_function_list.append(cdf)
            y = np.real(cdf(x))
            self._logger.info(f"cdf({x}) = {y}")
            if y > cutoff:
                c += 1
        if c <= n_batch // 2:
            b += 1

        self._logger.info(f"certify result: {b}")
        return bool(b)


class LT22GSEE(LT22PhaseEstimation[StateT]):
    def __init__(
        self,
        time_evo_estimator: TimeEvolutionExpectationValueEstimator[StateT],
        tau: float,
    ):
        unitary_power_estimator = TimeEvolutionPowerEstimator(time_evo_estimator, tau)
        super().__init__(unitary_power_estimator)


class SingleSignalLT22PhaseEstimation(StatisticalPhaseEstimation[StateT]):
    r"""The LT22 statistical phase estimation algorithm.

    Ref:
        Lin Lin and Yu Tong,
        Heisenberg-Limited Ground-State Energy Estimation for Early Fault-Tolerant
        Quantum Computers,
        PRX Quantum 3, 010318

    Args:
        unitary_power_estimator: Any estimator that estimates the expectation value
            :math:`\langle U^k \rangle`, where :math:`U` is a unitary operator and `k`
            is the power.
    """

    def __init__(
        self, unitary_power_estimator: UnitaryOpPowerSamplingEstimator[StateT]
    ):
        self._unitary_power_estimator = unitary_power_estimator

    @property
    def unitary_power_estimator(self) -> UnitaryOpPowerSamplingEstimator[StateT]:
        return self._unitary_power_estimator

    @staticmethod
    def invert_signal(signal_function: SPEDiscreteSignalFunction, eta: float) -> float:
        """Invert CDF algorithm of PRX Quantum 3, 010318 with a fixed signal.

        Args:
            signal_function: The signal function to be inverted.
            eta: The guess of the ground state overlap.
        """
        x0, x1 = -np.pi / 3, np.pi / 3
        cutoff = REF_OVERLAP_RATIO * eta
        while x1 - x0 > MIN_WIDTH:
            curr_x = (x0 + x1) / 2
            if np.real(signal_function(curr_x)) < cutoff:
                x0 = curr_x - 2 * eta / 3
            else:
                x1 = curr_x + 2 * eta / 3
            next_x = (x0 + x1) / 2
            if abs(curr_x - next_x) < CONVERGENCE_TOL:
                break
        return curr_x

    def __call__(
        self,
        state: StateT,
        step_function_param: "StepFunctionParam",
        eta: float,
        *,
        signal_function: Optional[SPEDiscreteSignalFunction] = None,
    ) -> SPEResult:
        """Invert CDF algorithm of PRX Quantum 3, 010318.

        Args:
            state: The input state to the phase estimation algorithm.
            step_function_param:
                A :class:`StepFunctionParam` for constructing the signal.
            eta: The guess of the ground state overlap.
            signal_function: A fixed signal function to be inverted.
                When this is passed in, other parameters except eta are ignored, i.e.,
                it is equivalent to running:
                >>> SingleSignalLT22PhaseEstimation.invert_signal(signal_funtion, eta)
        """
        if signal_function is None:
            signal_generator = StepFunctionSignalGenerator(
                self.unitary_power_estimator,
                state,
                step_function_param.d,
                step_function_param.delta,
            )
            signal_function = signal_generator(step_function_param.n_sample)
        phase = self.invert_signal(signal_function, eta)
        return SPEResult(phase=phase, signal_functions=[signal_function])


class SingleSignalLT22GSEE(SingleSignalLT22PhaseEstimation[StateT]):
    r"""Solves ground state energy estimation problem with
    :class:`SingleSignalLT22PhaseEstimation`.

    Args:
        time_evo_estimator: A :class:`TimeEvolutionExpectationValueEstimator`
            for estimating :math:`\langle e^{-ik\tau H}\rangle`, where :math: k
            is integer and :math:`\tau` is a dimensionless normalization factor.
        tau: The dimensionless normalization factor in
            :math:`\langle e^{-ik\tau H}\rangle`.
    """

    def __init__(
        self,
        time_evo_estimator: TimeEvolutionExpectationValueEstimator[StateT],
        tau: float,
    ):
        unitary_power_estimator = TimeEvolutionPowerEstimator(time_evo_estimator, tau)
        super().__init__(unitary_power_estimator)
