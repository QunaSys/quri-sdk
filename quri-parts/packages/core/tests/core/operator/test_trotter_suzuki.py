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

from quri_parts.core.operator import (
    PAULI_IDENTITY,
    Operator,
    pauli_label,
    trotter_suzuki_decomposition,
)
from quri_parts.core.operator.trotter_suzuki import ExponentialSinglePauli


def test_trotter_suzuki_decomposition() -> None:
    param = 0.1
    pauli_1 = pauli_label("X0 Y1 Z2 X3")
    pauli_2 = pauli_label("Z0 X1 Y2 Y3")
    op = Operator(
        {
            PAULI_IDENTITY: 2.9,
            pauli_1: 1.5,
            pauli_2: -1.3,
        }
    )

    with pytest.raises(ValueError):
        trotter_suzuki_decomposition(op, param, -1)

    s_2 = [
        ExponentialSinglePauli(PAULI_IDENTITY, param * 2.9 / 2),
        ExponentialSinglePauli(pauli_1, param * 1.5 / 2),
        ExponentialSinglePauli(pauli_2, param * -1.3 / 2),
        ExponentialSinglePauli(pauli_2, param * -1.3 / 2),
        ExponentialSinglePauli(pauli_1, param * 1.5 / 2),
        ExponentialSinglePauli(PAULI_IDENTITY, param * 2.9 / 2),
    ]
    assert trotter_suzuki_decomposition(op, param, 1) == s_2

    p_2 = 1 / (4 - 4 ** (1 / (2 * 2 - 1)))
    s_4_1 = [ExponentialSinglePauli(pn, cn * p_2) for pn, cn in s_2]
    s_4_2 = [ExponentialSinglePauli(pn, cn * (1 - 4 * p_2)) for pn, cn in s_2]

    s_4 = s_4_1 * 2 + s_4_2 + s_4_1 * 2
    exp_tsd = trotter_suzuki_decomposition(op, param, 2)
    for i, exp in enumerate(s_4):
        assert exp.pauli == exp_tsd[i].pauli
        assert np.isclose(exp_tsd[i].coefficient, exp.coefficient)
