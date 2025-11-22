# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import unittest.mock
from typing import NamedTuple

import numpy as np
from quri_parts.core.state import CircuitQuantumState, quantum_state

import quri_algo.algo.phase_estimation.spe as spe
from quri_algo.algo.phase_estimation.spe import SingleSignalLT22PhaseEstimation
from quri_algo.core.estimator import OperatorPowerEstimatorBase
from quri_algo.core.estimator.time_evolution import (
    TimeEvolutionExpectationValueEstimator,
)


class _Estimate(NamedTuple):
    value: complex
    error: float = np.nan


class TestLT22PhaseEstimation(unittest.TestCase):
    lt22_algo: SingleSignalLT22PhaseEstimation[CircuitQuantumState]

    @classmethod
    def setUpClass(cls) -> None:
        estimator = unittest.mock.Mock(
            spec=OperatorPowerEstimatorBase,
            return_value=_Estimate(value=1.0),
        )
        cls.lt22_algo = SingleSignalLT22PhaseEstimation(estimator)

    def test_lt22_phase_estimation(self) -> None:
        n_samples = int(1e10)
        d = 100
        delta = 1e-4
        eta = 0.8
        input_state = quantum_state(1)
        step_func_param = spe.StepFunctionParam(d, delta, n_samples)
        result = self.lt22_algo(input_state, step_func_param, eta)
        assert np.isclose(
            result.signal_functions[0](result.value.real), eta * 0.75, atol=1e-2
        )


class TestLT22GSEE(unittest.TestCase):
    lt22_algo: SingleSignalLT22PhaseEstimation[CircuitQuantumState]

    @classmethod
    def setUpClass(cls) -> None:
        time_evo_estimator = unittest.mock.Mock(
            spec=TimeEvolutionExpectationValueEstimator,
            return_value=_Estimate(value=1.0),
        )
        tau = 20
        cls.lt22_algo = spe.SingleSignalLT22GSEE(time_evo_estimator, tau)

    def test_lt22_gsee(self) -> None:
        input_state = quantum_state(1)
        n_samples = int(1e10)
        d = 100
        delta = 1e-4
        step_func_param = spe.StepFunctionParam(d, delta, n_samples)
        eta = 0.8

        result = self.lt22_algo(input_state, step_func_param, eta)
        assert np.isclose(
            result.signal_functions[0](result.value).real, eta * 0.75, atol=1e-2
        )
