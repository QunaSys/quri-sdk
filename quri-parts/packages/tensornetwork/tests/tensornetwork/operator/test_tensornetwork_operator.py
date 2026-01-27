# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_almost_equal

from quri_parts.core.operator import Operator, PauliLabel, pauli_label
from quri_parts.tensornetwork.operator import operator_to_tensor

operator_tensor_pairs: Sequence[
    tuple[Union[Operator, PauliLabel], npt.NDArray[np.complex128]]
] = [
    (
        Operator(
            {
                pauli_label("X0 Y1"): 1.0,
            }
        ),
        np.array(
            [
                [[[0.0, 0.0], [0.0, 1j]], [[0.0, 0.0], [-1j, 0.0]]],
                [[[0.0, 1j], [0.0, 0.0]], [[-1j, 0.0], [0.0, 0.0]]],
            ],
        ),
    ),
    (
        Operator(
            {
                pauli_label("Y1 X0"): 1.0,
            }
        ),
        np.array(
            [
                [[[0.0, 0.0], [0.0, 1j]], [[0.0, 0.0], [-1j, 0.0]]],
                [[[0.0, 1j], [0.0, 0.0]], [[-1j, 0.0], [0.0, 0.0]]],
            ],
        ),
    ),
    (
        pauli_label("Y1 X0"),
        np.array(
            [
                [[[0.0, 0.0], [0.0, 1j]], [[0.0, 0.0], [-1j, 0.0]]],
                [[[0.0, 1j], [0.0, 0.0]], [[-1j, 0.0], [0.0, 0.0]]],
            ],
        ),
    ),
]


def test_tensornetwork_operator() -> None:
    for o, t in operator_tensor_pairs:
        tensor = operator_to_tensor(o, convert_to_mpo=False)._container.pop()

        assert_almost_equal(tensor.tensor, t)
