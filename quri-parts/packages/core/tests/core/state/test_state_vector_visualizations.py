# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

try:
    import matplotlib  # noqa: F401
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - import guard
    pytest.skip("matplotlib not available", allow_module_level=True)

from quri_parts.core.state import QuantumStateVector


def test_qsphere_returns_figure() -> None:
    amp = 1 / np.sqrt(2)
    state = QuantumStateVector(2, vector=[amp, 0, 0, amp])

    fig = state.draw(output="qsphere")

    assert isinstance(fig, Figure)


def test_density_matrix_returns_figure() -> None:
    amp = 1 / np.sqrt(2)
    state = QuantumStateVector(1, vector=[amp, amp])

    fig = state.draw(output="density_matrix")

    assert isinstance(fig, Figure)
