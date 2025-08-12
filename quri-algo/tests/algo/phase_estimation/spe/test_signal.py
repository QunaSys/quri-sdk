# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
from typing import NamedTuple, Sequence, cast
from unittest.mock import Mock, patch

import numpy as np
import numpy.typing as npt
from quri_parts.core.state import CircuitQuantumState, quantum_state

import quri_algo.algo.phase_estimation.spe.utils.fourier as fourier
from quri_algo.algo.phase_estimation.spe.utils.exp_val_collector import (
    ExpectationValueCollector,
    SPESample,
    UnitaryOpPowerSamplingEstimator,
)
from quri_algo.algo.phase_estimation.spe.utils.signal import (
    GaussianSignalGenerator,
    SPEDiscreteSignalFunction,
    StepFunctionSignalGenerator,
)
from quri_algo.core.estimator import State


class _Estimate(NamedTuple):
    value: complex
    error: float = np.nan


class FakeSignalInfo:
    @property
    def n_fourier_modes(self) -> int:
        return 101

    @property
    def n_samples(self) -> int:
        return 1000 * self.n_fourier_modes

    @property
    def n_distribution_list(self) -> Sequence[int]:
        return list(np.ones(self.n_fourier_modes) * 1000)

    @property
    def classical_samples(self) -> Sequence[fourier.SPEFourierCoefficient]:
        return [
            fourier.SPEFourierCoefficient(k, n_dist, self.n_samples, 0.0)
            for k, n_dist in zip(
                range(-50, -50 + self.n_fourier_modes), self.n_distribution_list
            )
        ]

    @staticmethod
    def fake_unitary_power_estimator() -> UnitaryOpPowerSamplingEstimator[State]:
        return cast(
            UnitaryOpPowerSamplingEstimator[State],
            lambda state, operator_power, n_shots: _Estimate(value=1.0),
        )


def fake_random_sampler(
    n_samples: int, pvals: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return n_samples * np.array(pvals)


class TestStepFunctionSignalGenerator(unittest.TestCase):
    n_samples: int
    d: int
    delta: float
    expected_function: SPEDiscreteSignalFunction

    @classmethod
    def setUpClass(cls) -> None:
        """Fourier coefficient of another function are all 1."""
        cls.n_samples = int(1e10)
        cls.d = 10000
        cls.delta = 1e-4

        spe_samples, norm = cls.get_spe_samples_and_norm(
            cls.d, cls.delta, cls.n_samples
        )
        cls.expected_function = SPEDiscreteSignalFunction(spe_samples, norm_factor=norm)

    @staticmethod
    def get_spe_samples_and_norm(
        d: int, delta: float, n_sample: int
    ) -> tuple[Sequence[SPESample], float]:
        classical_sampler = fourier.StepFunctionSampler(d, delta)
        fourier_coeffs = classical_sampler.fourier_coefficients
        norm = np.linalg.norm(fourier_coeffs, 1)
        probs = np.abs(fourier_coeffs) / norm
        phases = np.angle(fourier_coeffs)

        spe_fourier_coeffs = [
            fourier.SPEFourierCoefficient(i - d, prob * n_sample, n_sample, phase)
            for i, (prob, phase) in enumerate(zip(probs, phases))
            if prob > 0
        ]
        return [
            SPESample(spe_fourier_coeff, 1.0 + 0.0j)
            for spe_fourier_coeff in spe_fourier_coeffs
        ], cast(float, norm)

    @patch(
        "quri_algo.algo.phase_estimation.spe.utils.fourier.step_func_coefficient.np.random"
    )
    def test_step_function_signal_generator(self, mock_numpy) -> None:  # type: ignore
        rng = Mock()
        rng.multinomial = fake_random_sampler
        mock_numpy.default_rng.return_value = rng

        estimator = cast(
            UnitaryOpPowerSamplingEstimator[State],
            lambda state, operator_power, n_shots: _Estimate(value=1.0),
        )

        state = quantum_state(100)
        generator: StepFunctionSignalGenerator[
            CircuitQuantumState
        ] = StepFunctionSignalGenerator(
            power_estimator=estimator, state=state, d=10000, delta=1e-4
        )
        signal_function = generator(n_sample=int(1e10))

        assert np.allclose(signal_function.ks, self.expected_function.ks)
        assert np.allclose(
            signal_function.conv_fourier_coeff,
            self.expected_function.conv_fourier_coeff,
        )
        assert np.allclose(
            signal_function.density_func_coeff,
            self.expected_function.density_func_coeff,
        )
        x = random.random()
        assert np.isclose(signal_function(x), self.expected_function(x))

        xs = np.array([random.random() for _ in range(10)])
        assert np.allclose(signal_function(xs), self.expected_function(xs))


class TestGaussianSignalGenerator(unittest.TestCase):
    n_samples: int
    cutoff_T: float
    n_discretize: int
    sigma: float
    expected_function: SPEDiscreteSignalFunction

    @classmethod
    def setUpClass(cls) -> None:
        """Fourier coefficient of another function are all 1."""
        cls.n_samples = int(1e6)
        cls.cutoff_T = 40.0
        cls.n_discretize = 100
        cls.sigma = 8.0

        spe_samples, norm = cls.get_spe_samples_and_norm(
            cls.cutoff_T, cls.n_discretize, cls.sigma, cls.n_samples
        )
        cls.expected_function = SPEDiscreteSignalFunction(
            spe_samples, norm_factor=norm, exponent_factor=2 * np.pi
        )

    @staticmethod
    def get_spe_samples_and_norm(
        cutoff_T: float, n_discretize: int, sigma: float, n_sample: int
    ) -> tuple[Sequence[SPESample], float]:
        classical_sampler = fourier.GaussianSampler(cutoff_T, n_discretize, sigma)
        fourier_coeffs = classical_sampler.fourier_coefficients
        norm = np.linalg.norm(fourier_coeffs, 1)
        probs = np.abs(fourier_coeffs) / norm
        phases = np.angle(fourier_coeffs)

        spe_fourier_coeffs = [
            fourier.SPEFourierCoefficient(
                -cutoff_T + 2 * cutoff_T / n_discretize * i,
                prob * n_sample,
                n_sample,
                phase,
            )
            for i, (prob, phase) in enumerate(zip(probs, phases))
            if prob > 0
        ]
        return [
            SPESample(spe_fourier_coeff, 1.0 + 0.0j)
            for spe_fourier_coeff in spe_fourier_coeffs
        ], cast(float, 2 * cutoff_T * norm / n_discretize)

    @patch(
        "quri_algo.algo.phase_estimation.spe.utils.fourier.gaussian_coefficient.np.random"
    )
    def test_gaussian_signal_generator(self, mock_numpy) -> None:  # type: ignore
        rng = Mock()
        rng.multinomial = fake_random_sampler
        mock_numpy.default_rng.return_value = rng

        estimator = cast(
            UnitaryOpPowerSamplingEstimator[State],
            lambda state, operator_power, n_shots: _Estimate(value=1.0),
        )

        state = quantum_state(100)
        generator: GaussianSignalGenerator[
            CircuitQuantumState
        ] = GaussianSignalGenerator(
            power_estimator=estimator,
            state=state,
            cutoff_T=self.cutoff_T,
            n_discretize=self.n_discretize,
            sigma=self.sigma,
        )
        signal_function = generator(n_sample=int(1e6))

        assert np.allclose(signal_function.ks, self.expected_function.ks)
        assert np.allclose(
            signal_function.conv_fourier_coeff,
            self.expected_function.conv_fourier_coeff,
        )
        assert np.allclose(
            signal_function.density_func_coeff,
            self.expected_function.density_func_coeff,
        )
        x = random.random()
        print(signal_function(x) / self.expected_function(x))
        assert np.isclose(signal_function(x), self.expected_function(x))

        xs = np.array([random.random() for _ in range(10)])
        assert np.allclose(signal_function(xs), self.expected_function(xs))


class TestExpectationValueCollector(unittest.TestCase):
    """test cases for :class:`ExpectationValueCollector`."""

    info: FakeSignalInfo

    def setUp(self) -> None:
        self.info = FakeSignalInfo()

    def test_expectation_value_collector(self) -> None:
        collector = ExpectationValueCollector(
            self.info.fake_unitary_power_estimator(), quantum_state(100)
        )
        spe_samples = collector(self.info.classical_samples)
        assert len(spe_samples) == self.info.n_fourier_modes
        for sample, _ in zip(spe_samples, range(-50, 50)):
            assert isinstance(sample, SPESample)
            assert sample.op_exp_val == 1.0


class TestSignalFunction(unittest.TestCase):
    info: FakeSignalInfo
    norm_factor: float
    exp_factor: float

    def setUp(self) -> None:
        self.info = FakeSignalInfo()
        self.norm_factor = 20
        self.exp_factor = 2 * np.pi

    def mimicked_function(
        self, x: float | npt.NDArray[np.float64]
    ) -> complex | npt.NDArray[np.complex128]:
        n_modes = self.info.n_fourier_modes
        n = n_modes // 2
        c = self.exp_factor
        num = np.exp(1j * c * n * x) * (1 - np.exp(-1j * c * n_modes * x))
        den = 1 - np.exp(-1j * c * x)
        return cast(
            complex | npt.NDArray[np.complex128], num / den / n_modes * self.norm_factor
        )

    def mimicked_1st_deriv_function(
        self, x: float | npt.NDArray[np.float64]
    ) -> complex | npt.NDArray[np.complex128]:
        n_modes = self.info.n_fourier_modes
        n = n_modes // 2
        c = self.exp_factor
        num = (np.exp(2j * c * (1 + n) * x) - 1) * n
        num += (n + 1) * np.exp(1j * c * x)
        num -= (n + 1) * np.exp(1j * c * (1 + 2 * n) * x)
        num = num * 1j * c * np.exp(-1j * c * n * x)
        den = (-1 + np.exp(1j * c * x)) ** 2
        return cast(
            complex | npt.NDArray[np.complex128], num / den / n_modes * self.norm_factor
        )

    def test_collector(self) -> None:
        collector = ExpectationValueCollector(
            self.info.fake_unitary_power_estimator(), quantum_state(100)
        )
        spe_samples = collector(self.info.classical_samples)
        signal_function = SPEDiscreteSignalFunction(
            spe_samples, self.norm_factor, exponent_factor=self.exp_factor
        )
        x = np.random.random()
        assert np.isclose(signal_function(x), self.mimicked_function(x))
        assert np.isclose(signal_function(x, 1), self.mimicked_1st_deriv_function(x))

        xs = np.random.random(100)
        assert np.allclose(signal_function(xs), self.mimicked_function(xs))
        assert np.allclose(signal_function(xs, 1), self.mimicked_1st_deriv_function(xs))
