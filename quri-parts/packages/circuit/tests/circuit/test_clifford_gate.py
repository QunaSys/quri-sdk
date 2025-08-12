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

from quri_parts.circuit import (
    CNOT,
    RX,
    RY,
    U2,
    U3,
    H,
    Pauli,
    PauliRotation,
    T,
    X,
    Y,
    is_clifford,
)


def test_is_clifford() -> None:
    assert is_clifford(X(0))
    assert is_clifford(Y(1))
    assert is_clifford(H(2))
    assert is_clifford(T(0)) is False
    assert is_clifford(RX(0, 0.5 * np.pi))
    assert is_clifford(RY(0, 0.8 * np.pi)) is False
    assert is_clifford(U2(0, 0.5 * np.pi, 1.7 * np.pi)) is False
    assert is_clifford(U3(0, 0.5 * np.pi, 1.5 * np.pi, 2.5 * np.pi))
    assert is_clifford(CNOT(0, 1))
    assert is_clifford(Pauli([0, 1, 2], [1, 2, 0]))
    assert is_clifford(PauliRotation([0, 1, 2], [1, 2, 0], 1.5 * np.pi))
    assert is_clifford(PauliRotation([0, 1, 2], [1, 2, 0], 0.51 * np.pi)) is False
