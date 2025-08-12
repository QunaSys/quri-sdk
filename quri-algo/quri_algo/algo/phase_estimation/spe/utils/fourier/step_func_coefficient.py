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
from typing import Sequence, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy import integrate

from .coefficient import FourierCoefficientSampler, SPEFourierCoefficient

NumpyInput = TypeVar("NumpyInput", float, np.float64, npt.NDArray[np.float64])


@dataclass
class StepFunctionParam:
    r"""The parameters for building the step function convolution function.

    Args:
        d: The maximal value of the Fourier mode k.
        delta: The parameter that controls the precision:

            .. math::
                |F_{d, \delta}(x) - H(x)| \leq \epsilon \quad \forall x \in [-\pi + \delta, - \delta] \cup [\delta, \pi - \delta]

        n_sample:
            Number of samples to construct the step function.
    """

    d: int
    delta: float
    n_sample: int


def get_Tn(x: NumpyInput, n: int) -> NumpyInput:
    # x is defined in [-pi, pi]!!
    coef = np.zeros(n + 1)
    coef[n] = 1
    return cast(NumpyInput, np.polynomial.chebyshev.chebval(x=x, c=coef))


_normalization_cache: dict[tuple[int, float], float] = {}


def get_M(x: NumpyInput, d: int, delta: float, normalize: bool = False) -> NumpyInput:
    def integrand(x: NumpyInput, d: int, delta: float) -> NumpyInput:
        arg = 1 + 2 * (np.cos(x) - np.cos(delta)) / (1 + np.cos(delta))
        return cast(NumpyInput, get_Tn(x=arg, n=d))

    if normalize:
        if (d, delta) in _normalization_cache:
            N = _normalization_cache[(d, delta)]
        else:
            N, _ = integrate.quad(integrand, -np.pi, np.pi, args=(d, delta))
            _normalization_cache[(d, delta)] = N
    else:
        N = 1

    return integrand(x, d, delta) / N


def get_M_tilde(
    d: int, delta: float, normalize: bool = False
) -> npt.NDArray[np.complex128]:
    r"""Riemann sum of Fourier transform integral.

    .. math::
        M_{d, \delta, k} =
            \frac{1}{\sqrt{2 \pi}} (-1)^k \sum_{n=0}^{N - 1}
            M_{d, \delta}(-\pi + \frac{2 n \pi}{N})
            e^{-\frac{2n\pi}{N}i} \frac{2\pi}{N}
    """
    N = 2 * d
    D = 2 * np.pi / N
    x = np.linspace(-np.pi, np.pi - D, N, dtype=np.float64)
    x_ft = np.fft.fft(get_M(x, d, delta, normalize))

    for i, _ in enumerate(x_ft):
        x_ft[i] *= (-1) ** i

    return x_ft / cast(np.float64, np.sqrt(2 * np.pi)) * D


def get_H_tilde(k: int) -> complex:
    if k % 2 == 0 and k != 0:
        return 0
    if k == 0:
        return cast(complex, np.sqrt(np.pi / 2))
    if k % 2 != 0:
        return -1j * 2 / cast(complex, np.sqrt(2 * np.pi) * k)
    raise ValueError


def _get_minus_k_F_tilde_coefficient(
    half_fourier_coefficient: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    left_half = half_fourier_coefficient[1:][::-1].conj()
    return np.hstack([left_half, half_fourier_coefficient])


_f_tilde_cache: dict[tuple[int, float], npt.NDArray[np.complex128]] = {}


def get_F_tilde(d: int, delta: float) -> npt.NDArray[np.complex128]:
    if (d, delta) in _f_tilde_cache:
        return _f_tilde_cache[(d, delta)].copy()

    M_fourier = get_M_tilde(d, delta)
    M_fourier = M_fourier[0 : d + 1]
    H_fourier = np.array([get_H_tilde(k) for k in range(d + 1)])

    normalized_coeff = M_fourier * H_fourier
    normalized_coeff = normalized_coeff / normalized_coeff[0] * 0.5
    normalized_coeff = _get_minus_k_F_tilde_coefficient(normalized_coeff)
    _f_tilde_cache[(d, delta)] = normalized_coeff.copy()
    return cast(npt.NDArray[np.complex128], normalized_coeff)


def get_k_from_idx(idx: int, d: int) -> int:
    r""":math:`F_{k}` is labelled as

    .. math::
        [F_{-d}, ..., F_{-1}, F_{0}, F_{1}, ..., F_{d}]
    """
    return idx - d


def sample_from_F_tilde(distribution: Sequence[complex], n_sample: int) -> Counter[int]:
    """index :math:`i` represents :math:`F_{k=-d+i}`."""
    if np.sum(np.abs(distribution)) != 1:
        distribution /= np.sum(np.abs(distribution))
    random_gen = np.random.default_rng()
    sample = random_gen.multinomial(n_sample, pvals=np.abs(distribution))
    counter = Counter({i: c for i, c in enumerate(sample) if c > 0})
    return counter


class StepFunctionSampler(FourierCoefficientSampler):
    """The sampler that samples from the Fourier coefficients of the step
    function.

    Args:
        d: Number of Fourier coefficients.
        delta: The precision of the the Fourier transform. Check out
            Lemma 5 of PRX Quantum 3, 010318 for precise definition.
    """

    def __init__(self, d: int, delta: float):
        self.d = d
        self.delta = delta

    @property
    def fourier_coefficients(self) -> npt.NDArray[np.complex128]:
        r"""The Fourier coefficients of the periodic step function.

        .. math::     \left[         \widetilde{F}(-d),
        \widetilde{F}(-d+1), \cdots, \widetilde{F}(d-1),
        \widetilde{F}(d)     \right]
        """
        return get_F_tilde(self.d, self.delta)

    def get_phase(self, k: int) -> float:
        """Get phase of the k-th Fourier coefficient."""
        return cast(float, np.angle(self.fourier_coefficients[k + self.d]))

    def __call__(self, n_samples: int) -> Sequence[SPEFourierCoefficient]:
        classical_samples = sample_from_F_tilde(
            self.fourier_coefficients.tolist(), n_samples
        )
        f_coeff_list: list[SPEFourierCoefficient] = []
        for i, cnt in classical_samples.items():
            k = get_k_from_idx(i, self.d)
            fourier_coeff = SPEFourierCoefficient(k, cnt, n_samples, self.get_phase(k))
            f_coeff_list.append(fourier_coeff)
        return f_coeff_list
