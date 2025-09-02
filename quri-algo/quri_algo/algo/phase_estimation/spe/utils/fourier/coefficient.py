# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import NamedTuple, Sequence, cast

import numpy as np
import numpy.typing as npt


class SPEFourierCoefficient(NamedTuple):
    """Fourier coefficient for SPE.

    .. math::

        \\tilde{F}(k_n)
            = \\mathcal{F} \\frac{|\\tilde{F}(k_n)|}{\\mathcal{F}} e^{i \\varphi_{kn}}
            = \\mathcal{F} N_n/N_{\\text{sample}} e^{i \\varphi_{k_n}}
    Args:
        k: The label of the Fourier mode.
        n_distribute: Number of samples distributed to this Fourier mode.
        n_total_sample: Total number of samples used to sample the Fourier coeffients.
        phase: The phase of the Fourier coefficient.
    """

    k: int | float
    n_distribute: int
    n_total_sample: int
    phase: float

    def get_coeff(self) -> complex:
        magnitude = self.n_distribute / self.n_total_sample
        return cast(complex, magnitude * np.exp(1j * self.phase))


class FourierCoefficientSampler(ABC):
    """Sampler that samples from the discrete Fourier coefficients."""

    @property
    @abstractmethod
    def fourier_coefficients(self) -> npt.NDArray[np.complex128]:
        """The Fourier coefficients.

        The coefficient of the smallest Fourier mode k will occupy the
        0-th entry of the array.
        """

    @abstractmethod
    def get_phase(self, k: int) -> float:
        """Get phase of the k-th Fourier coefficient."""

    @abstractmethod
    def __call__(self, n_samples: int) -> Sequence[SPEFourierCoefficient]:
        ...
