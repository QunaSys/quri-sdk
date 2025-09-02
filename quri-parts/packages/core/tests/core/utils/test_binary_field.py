# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.core.utils.binary_field import (
    BinaryArray,
    BinaryMatrix,
    hstack,
    inverse,
    vstack,
)


class TestBinaryArray:
    def test_len(self) -> None:
        assert len(BinaryArray([0, 1])) == 2
        assert len(BinaryArray([0, 1, 0])) == 3
        assert len(BinaryArray([])) == 0

    def test_bitwise_xor(self) -> None:
        assert BinaryArray([0, 0, 1, 1]) + BinaryArray([0, 1, 0, 1]) == BinaryArray(
            [0, 1, 1, 0]
        )

    def test_bitwise_and(self) -> None:
        assert BinaryArray([0, 0, 1, 1]) * BinaryArray([0, 1, 0, 1]) == BinaryArray(
            [0, 0, 0, 1]
        )

    def test_getitem_and_setitem(self) -> None:
        ba = BinaryArray([0, 0, 0])
        assert ba[2] == 0
        ba[2] = 1
        assert ba[2] == 1

    def test_equal(self) -> None:
        assert BinaryArray([0, 1]) != BinaryArray([1, 0])
        assert BinaryArray([0, 1]) == BinaryArray([0, 1])

    def test_inner_product(self) -> None:
        assert BinaryArray([1, 0, 1, 0, 1]) @ BinaryArray([1, 1, 0, 0, 1]) == 0
        assert BinaryArray([1, 1, 1, 0, 1]) @ BinaryArray([1, 1, 0, 0, 1]) == 1


class TestBinaryMatrix:
    def test_len(self) -> None:
        assert len(BinaryMatrix([[0, 1], [1, 1]])) == 2
        assert len(BinaryMatrix([[0, 1], [0, 0], [1, 1]])) == 3
        assert len(BinaryMatrix([])) == 0

    def test_invalid_shape_raises_error(self) -> None:
        with pytest.raises(ValueError):
            BinaryMatrix([[0, 1], [1, 1, 1]])

    def test_getitem_and_setitem(self) -> None:
        bm = BinaryMatrix([[0, 1], [1, 0]])
        assert bm[0] == BinaryArray([0, 1])
        assert bm[1] == BinaryArray([1, 0])
        bm[1] = BinaryArray([1, 1])
        assert bm[1] == BinaryArray([1, 1])
        bm[0, 1] = 0
        assert bm[0, 1] == 0

    def test_matmul_with_array(self) -> None:
        assert BinaryMatrix([[0, 1], [1, 0]]) @ BinaryArray([0, 1]) == BinaryArray(
            [1, 0]
        )

    def test_matmul_with_matrix(self) -> None:
        assert BinaryMatrix([[0, 1], [1, 0]]) @ BinaryMatrix(
            [[0, 1], [1, 0]]
        ) == BinaryMatrix([[1, 0], [0, 1]])

    def test_equal(self) -> None:
        assert BinaryMatrix([[0, 1], [0, 0]]) != BinaryMatrix([[1, 0], [0, 0]])
        assert BinaryMatrix([[0, 1], [0, 0]]) == BinaryMatrix([[0, 1], [0, 0]])

    def test_transpose(self) -> None:
        assert BinaryMatrix([[1, 0, 1], [0, 0, 1]]).transpose() == BinaryMatrix(
            [[1, 0], [0, 0], [1, 1]]
        )


def test_hstack() -> None:
    a = BinaryMatrix([[0, 1], [1, 1]])
    b = BinaryMatrix([[1, 0], [0, 0]])
    expected = BinaryMatrix([[0, 1, 1, 0], [1, 1, 0, 0]])
    assert hstack(a, b) == expected


def test_vstack() -> None:
    a = BinaryMatrix([[0, 1], [1, 1]])
    b = BinaryMatrix([[1, 0], [0, 0]])
    expected = BinaryMatrix([[0, 1], [1, 1], [1, 0], [0, 0]])
    assert vstack(a, b) == expected


def test_inverse() -> None:
    a = BinaryMatrix([[0, 1], [1, 0]])
    assert inverse(a) == a

    b = BinaryMatrix([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    b_inv = inverse(b)
    assert b @ b_inv == BinaryMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
