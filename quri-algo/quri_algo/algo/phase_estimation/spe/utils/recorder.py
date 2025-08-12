# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .signal import SPEDiscreteSignalFunction


@dataclass
class SPEGSEERecorder:
    r"""Compute :math:`T_{\text{max}}`, :math:`T_{\text{total}}` and
    :math:`N_{\text{shot}}` of a ground state estimation algorthm.

    Args:
        signal_functions: A sequence of :class:`~SPEDiscreteSignalFunction`.
        norm_factor: The :math:`\tau` normalization factor.
    """

    signal_functions: Sequence[SPEDiscreteSignalFunction] = field(init=True, repr=False)
    norm_factor: float = field(init=True, repr=False)
    max_evolution_time: float = field(init=False)
    total_evolution_time: float = field(init=False)
    n_shots: int = field(init=False)

    def __post_init__(self) -> None:
        (
            self.max_evolution_time,
            self.total_evolution_time,
            self.n_shots,
        ) = self._compute_indicators()

    def _compute_indicators(self) -> tuple[float, float, int]:
        t_max, t_total, n_shots = 0.0, 0.0, 0
        for signal in self.signal_functions:
            for sample in signal.spe_samples:
                evo_time = np.abs(sample.classical_sample.k * self.norm_factor)
                rep = sample.classical_sample.n_distribute
                t_max = np.max([t_max, evo_time])
                t_total += 2 * evo_time * rep
                n_shots += 2 * rep
        return t_max, t_total, n_shots
