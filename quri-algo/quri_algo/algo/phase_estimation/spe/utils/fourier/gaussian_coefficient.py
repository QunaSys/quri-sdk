# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
import numpy.typing as npt

from .coefficient import FourierCoefficientSampler, SPEFourierCoefficient


@dataclass
class GaussianParam:
    r"""Hyperparameters of the Gaussian distribution function.

    .. math::
        G_{\sigma}(x) = \int_{-T}^{T} e^{-\frac{1}{2}(\sigma^2 \pi^2 k^2)} e^{2\pi i k x} dk

    Args:
        T: The boundary of the Fourier integration region.
        N: Number of points to disceretize the integration interval.
        sigma: Standard deviation of the Gaussian distribution.
        n_sample: Number of samples to sample the Fourier coefficient
            :math:`e^{-\frac{1}{2}(\sigma^2 \pi^2 k^2)}`
    """

    T: float
    N: int
    sigma: float
    n_sample: int


def get_kn(T: float, N: int) -> npt.NDArray[np.float64]:
    n = np.arange(N)
    return -T + 2 * T / N * n


def get_classical_samples(distributions: list[float], n_samples: int) -> Counter[int]:
    rng = np.random.default_rng()
    samples = rng.multinomial(n_samples, distributions)
    return Counter({i: n for i, n in enumerate(samples) if n > 0})


class GaussianSampler(FourierCoefficientSampler):
    def __init__(
        self,
        cutoff_T: float,
        n_discretize: int,
        sigma: float,
    ):
        self.cutoff_T = cutoff_T
        self.n_discretize = n_discretize
        self.sigma = sigma

    @property
    def kn(self) -> npt.NDArray[np.float64]:
        return get_kn(self.cutoff_T, self.n_discretize)

    @property
    def fourier_coefficients(self) -> npt.NDArray[np.complex128]:
        kn = get_kn(self.cutoff_T, self.n_discretize)
        return cast(
            npt.NDArray[np.complex128], np.exp(-2 * (np.pi * kn * self.sigma) ** 2)
        )

    def get_phase(self, k: int) -> float:
        return 0.0

    @property
    def distribution(self) -> npt.NDArray[np.float64]:
        return self.fourier_coefficients / np.linalg.norm(self.fourier_coefficients, 1)

    def __call__(self, n_samples: int) -> Sequence[SPEFourierCoefficient]:
        classical_samples = get_classical_samples(self.distribution.tolist(), n_samples)
        f_coeff_list: list[SPEFourierCoefficient] = []
        for i, cnt in classical_samples.items():
            k = self.kn[i]
            fourier_coeff = SPEFourierCoefficient(k, cnt, n_samples, 0.0)
            f_coeff_list.append(fourier_coeff)
        return f_coeff_list
