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

from quri_parts.core.operator import PAULI_IDENTITY, Operator, PauliLabel, pauli_label
from quri_parts.tensornetwork.operator import (
    operator_to_tensor,
    operator_to_tensor_sequence,
)

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


_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, 1j], [-1j, 0.0]], dtype=np.complex128)

identity_tensor_pairs: Sequence[
    tuple[Union[Operator, PauliLabel], npt.NDArray[np.complex128]]
] = [
    (PAULI_IDENTITY, np.eye(2, dtype=np.complex128)),
    (Operator({PAULI_IDENTITY: 2.0}), 2.0 * np.eye(2, dtype=np.complex128)),
]


def test_tensornetwork_operator_identity() -> None:
    for convert_to_mpo in (False, True):
        for o, t in identity_tensor_pairs:
            tensor_operator = operator_to_tensor(o, convert_to_mpo=convert_to_mpo)
            assert tensor_operator.index_list == [0]
            tensor = tensor_operator._container.pop()
            assert_almost_equal(tensor.tensor, t)


def test_tensornetwork_operator_sequence() -> None:
    (tensor_operator,) = operator_to_tensor_sequence(
        Operator({pauli_label("X0 Y1"): 2.0})
    )
    assert tensor_operator.index_list == [0, 1]
    tensor_map = tensor_operator.layer_tensor_map[0]
    assert_almost_equal(
        np.kron(tensor_map[1].tensor, tensor_map[0].tensor), 2.0 * np.kron(_Y, _X)
    )


def test_tensornetwork_operator_sequence_identity() -> None:
    for o, t in identity_tensor_pairs:
        (tensor_operator,) = operator_to_tensor_sequence(o)
        assert tensor_operator.index_list == [0]
        assert_almost_equal(tensor_operator.layer_tensor_map[0][0].tensor, t)
