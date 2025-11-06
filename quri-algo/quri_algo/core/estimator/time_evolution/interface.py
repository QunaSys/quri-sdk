# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Optional, Protocol, Union

from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.core.estimator import Estimate
from quri_parts.core.sampling import Sampler, StateSampler

from quri_algo.circuit.hadamard_test import HadamardTestCircuitFactory
from quri_algo.circuit.time_evolution.interface import (
    ControlledTimeEvolutionCircuitFactory,
)
from quri_algo.core.estimator import (
    ExpectationValueEstimator,
    OperatorPowerEstimatorBase,
    StateT,
)
from quri_algo.core.estimator.hadamard_test import HadamardTest
from quri_algo.problem import HamiltonianT


class TimeEvolutionExpectationValueEstimator(
    ExpectationValueEstimator[StateT], Protocol
):
    r"""Estimator that computes :math:`\langle e^{-iHt} \rangle`"""

    @abstractmethod
    def __call__(
        self, state: StateT, evolution_time: float, n_shots: Optional[int] = None
    ) -> Estimate[complex]:
        ...


class TimeEvolutionHadamardTest(TimeEvolutionExpectationValueEstimator[StateT]):
    r"""Computes :math:`\langle e^{-iHt} \rangle` with Hadamard test.

    This object can be instantiated or inherited for performing the
    estimation with Hadamard test for arbitrary implementations of the
    time evolution operator.
    """

    def __init__(
        self,
        encoded_problem: HamiltonianT,
        controlled_time_evolution_factory: ControlledTimeEvolutionCircuitFactory,
        sampler: Union[Sampler, StateSampler[StateT]],
        *,
        transpiler: CircuitTranspiler | None = None
    ):
        self.encoded_problem = encoded_problem
        self.controlled_time_evolution_factory = controlled_time_evolution_factory
        self.sampler = sampler
        self.transpiler = transpiler

        self._hadamard_test = HadamardTest(
            self.controlled_time_evolution_factory,
            self.sampler,
            transpiler=self.transpiler,
        )

    @property
    def real_circuit_factory(self) -> HadamardTestCircuitFactory:
        return self._hadamard_test.real_circuit_factory

    @property
    def imag_circuit_factory(self) -> HadamardTestCircuitFactory:
        return self._hadamard_test.imag_circuit_factory

    def __call__(
        self, state: StateT, evolution_time: float, n_shots: Optional[int] = None
    ) -> Estimate[complex]:
        assert n_shots is not None
        return self._hadamard_test(state, n_shots, evolution_time)


class TimeEvolutionPowerEstimator(OperatorPowerEstimatorBase[StateT]):
    r"""Turns a :class:`TimeEvolutionExpectationValueEstimator` for evaluating
    :math:`e^{-iH k \tau}` into a  :class:`OperatorPowerEstimatorBase` so that
    :math:`e^{-iH\tau}` is treated as the unitary operator and :math:`k` as the power.
    """

    def __init__(
        self,
        time_evo_estimator: TimeEvolutionExpectationValueEstimator[StateT],
        tau: float,
    ):
        self.time_evo_estimator = time_evo_estimator
        self.tau = tau

    def __call__(
        self, state: StateT, operator_power: int | float, n_shots: Optional[int] = None
    ) -> Estimate[complex]:
        return self.time_evo_estimator(state, operator_power * self.tau, n_shots)
