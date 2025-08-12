# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np

from quri_algo.algo.phase_estimation.spe.utils.fourier.gaussian_coefficient import (
    GaussianParam,
)


def get_gaussian_integration_bound(
    sigma: float, target_eps: float, overlap: float
) -> float:
    """Compute the integration bound :math:`T` of the convolution integration
    region."""
    eps_tilde = 0.1 * target_eps * overlap / (np.sqrt(2 * np.pi) * sigma**3)
    return float(
        np.pi**-1
        * sigma**-1
        * np.sqrt(2 * np.log(8 * np.pi**-1 * eps_tilde**-1 * sigma**-2))
    )


def get_gaussian_sigma(gap: float, target_eps: float, overlap: float) -> float:
    r"""Compute the standard deviation :math:`\sigma`."""
    return float(
        np.min(
            [
                0.9
                * gap
                / np.sqrt(2 * np.log(9 * gap * target_eps**-1 * overlap**-1)),
                0.2 * gap,
            ]
        )
    )


def get_number_of_search_points(sigma: float, target_eps: float) -> int:
    """Compute the number of points to search for the eigenvalue."""
    return int(np.ceil(sigma / target_eps) + 1)


def get_fourier_coeff_1_norm(
    sigma: float, n_discretize: int, integration_bound: float
) -> float:
    """Compute the 1-norm of the fourier coeffcients."""
    from quri_algo.algo.phase_estimation.spe.utils.fourier import GaussianSampler

    fourier_coeffs = GaussianSampler(
        integration_bound, n_discretize, sigma
    ).fourier_coefficients
    return float(np.sum(np.abs(2 * integration_bound / n_discretize * fourier_coeffs)))


def get_number_of_samples(
    sigma: float,
    target_eps: float,
    integration_bound: float,
    delta: float,
    n_search_points: int,
    n_discretize: int,
    sample_const: float = 2 * np.pi,
) -> int:
    shot_const = get_fourier_coeff_1_norm(sigma, n_discretize, integration_bound)
    return int(
        sample_const
        * sigma**4
        * shot_const**2
        * np.ceil(np.log(4 * n_search_points / delta) / target_eps**2)
    )


def get_recommended_gaussian_parameter(
    gap: float,
    target_eps: float,
    overlap: float,
    delta: float,
    n_discretize: int,
    tau: float = 1.0,
    sample_const: float = 2 * np.pi,
    max_shot_limit: Optional[int] = None,
) -> tuple[GaussianParam, int]:
    r"""The recommended parameters to execute the Gaussian SPE.

    Reference:
    Guoming Wang, Daniel Stilck França, Ruizhe Zhang, Shuchen Zhu, Peter D. Johnson
        Quantum algorithm for ground state energy estimation using circuit depth with
        exponentially improved dependence on precision
        Quantum 7, 1167 (2023).

    Args:
        gap: The (approximate) energy gap between the ground state the 1st excited state.
        eps: The target accuracy one wants to reach with Gaussian SPE.
        overlap: The (approximate) overlap of the input state and the exact ground state.
        delta: The tolerable failure probability of the Gaussian SPE.
        n_discretize: Number of discretization steps one wants to use to perform the
            convolution integration. This also corresponds to the number of evolution time
            steps one can sample from.
        tau: normalization factor of the spectrum
        n_sample_const: An overall coefficient :math:`C` for determining the number of samples.
            :math:`C \sigma^4 ||\\tilde{f}_T||_1 / \epsion^2 \\times \log`
        max_shot_limit: the maximal number of shots.

    Returns:
        GaussianParam: A :class:`GaussianParam` that specifies the Gaussian distribution function.
        int: The number of points to search for the eigenvalue.
    """

    gap = gap * tau
    target_eps = target_eps * tau

    sigma = get_gaussian_sigma(gap, target_eps, overlap)
    M = get_number_of_search_points(sigma, target_eps)
    T = get_gaussian_integration_bound(sigma, target_eps, overlap)
    S = get_number_of_samples(
        sigma, target_eps, T, delta, M, n_discretize, sample_const
    )
    if max_shot_limit is not None:
        S = np.min([S, max_shot_limit])
    return GaussianParam(T, n_discretize, sigma, S), M
