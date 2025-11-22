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
from quri_algo.algo.phase_estimation.spe.gaussian import (
    get_recommended_gaussian_parameter,
)
from quri_algo.core.estimator import OperatorPowerEstimatorBase
from quri_algo.core.estimator.time_evolution import (
    TimeEvolutionExpectationValueEstimator,
)


class _Estimate(NamedTuple):
    value: complex
    error: float = np.nan


class TestGaussianPhaseEstimation(unittest.TestCase):
    gaussian_algo: spe.GaussianFittingPhaseEstimation[CircuitQuantumState]
    gaussian_param: spe.GaussianParam
    a: float

    @classmethod
    def setUpClass(cls) -> None:
        int_boundary = 100
        n_discretize = 10000
        sigma = 1 / (np.sqrt(2 * np.pi))
        n_sample = 10000
        cls.gaussian_param = spe.GaussianParam(
            int_boundary, n_discretize, sigma, n_sample
        )

        cls.a = np.pi / 4

        estimator = unittest.mock.Mock(
            spec=OperatorPowerEstimatorBase,
            side_effect=lambda state, operator_power, n_shots: _Estimate(
                value=np.exp(-1j * operator_power * 2 * np.pi * cls.a)
            ),
        )
        cls.gaussian_algo = spe.GaussianFittingPhaseEstimation(estimator)

    def test_gaussian(self) -> None:
        search_range = np.linspace(self.a - np.pi * 3, self.a + np.pi * 3, 10000)
        input_state = quantum_state(1)
        result = self.gaussian_algo(input_state, self.gaussian_param, search_range, 0.8)
        assert np.isclose(result.value, self.a, atol=1e-8)


class TestGaussianGSEE(unittest.TestCase):
    gaussian_algo: spe.GaussianFittingPhaseEstimation[CircuitQuantumState]
    gaussian_param: spe.GaussianParam
    shift: float
    tau: float

    @classmethod
    def setUpClass(cls) -> None:
        int_boundary = 100
        n_discretize = 10000
        sigma = 1 / (np.sqrt(2 * np.pi))
        n_sample = 10000
        cls.gaussian_param = spe.GaussianParam(
            int_boundary, n_discretize, sigma, n_sample
        )
        cls.shift = np.pi / 4

        time_evo_estimator = unittest.mock.Mock(
            spec=TimeEvolutionExpectationValueEstimator,
            side_effect=lambda state, evolution_time, n_shots: _Estimate(
                value=np.exp(-1j * evolution_time * cls.shift)
            ),
        )

        cls.tau = 1 / 20
        cls.gaussian_algo = spe.GaussianFittingGSEE(time_evo_estimator, cls.tau)

    def test_gaussian_gsee(self) -> None:
        search_range = np.linspace(
            self.shift - np.pi * 3, self.shift + np.pi * 3, 10000
        )
        input_state = quantum_state(1)
        result = self.gaussian_algo(input_state, self.gaussian_param, search_range, 0.8)
        assert np.isclose(result.value, self.shift * self.tau, atol=1e-8)


def test_get_recommended_param() -> None:
    gap = 1.0085
    target_eps = 1e-3
    overlap = 0.96
    delta = 1e-4
    n_discretize = 1000
    param, n_search_pts = get_recommended_gaussian_parameter(
        gap, target_eps, overlap, delta, n_discretize
    )

    expected_sigma = 0.2 * gap
    ft_norm = 1 / (np.sqrt(2 * np.pi) * expected_sigma)
    expected_M = int(np.ceil(expected_sigma / target_eps) + 1)
    expected_n_sample = int(
        2
        * np.pi
        * expected_sigma**4
        * ft_norm**2
        * np.ceil(np.log(4 * expected_M / delta) / target_eps**2)
    )
    assert expected_sigma == param.sigma
    assert np.isclose(expected_n_sample, param.n_sample)
    assert np.isclose(param.N, n_discretize)
    assert expected_M == n_search_pts


def test_get_rescaled_recommended_param() -> None:
    gap = 1.0085
    target_eps = 1e-3
    overlap = 0.96
    delta = 1e-4
    n_discretize = 1000
    tau = 1 / 20
    expected_param, expected_n_search_pts = get_recommended_gaussian_parameter(
        tau * gap, tau * target_eps, overlap, delta, n_discretize
    )

    rescaled_param, rescaled_n_search_pts = get_recommended_gaussian_parameter(
        gap, target_eps, overlap, delta, n_discretize, tau
    )

    assert expected_param == rescaled_param
    assert expected_n_search_pts == rescaled_n_search_pts
