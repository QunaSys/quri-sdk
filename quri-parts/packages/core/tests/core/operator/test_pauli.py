# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch
from weakref import WeakValueDictionary

import pytest

from quri_parts.core.operator import (
    PAULI_IDENTITY,
    PauliLabel,
    SinglePauli,
    pauli_label,
    pauli_name,
    pauli_product,
)


def test_pauli_name() -> None:
    assert pauli_name(1) == "X"
    assert pauli_name(2) == "Y"
    assert pauli_name(3) == "Z"


class TestPauliLabel:
    def test_construct_from_tuples(self) -> None:
        for label in (
            PauliLabel(((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))),
            pauli_label(((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))),
        ):
            assert len(label) == 3
            assert label.pauli_at(2) == SinglePauli.Y
            assert label.pauli_at(6) == SinglePauli.Z
            assert label.pauli_at(4) == SinglePauli.X

    def test_construct_from_str(self) -> None:
        """Construct from a string without redundant spaces: 'Y2 Z6 X4'."""
        for label in (
            PauliLabel.from_str("Y2 Z6 X4"),
            pauli_label("Y2 Z6 X4"),
        ):
            assert len(label) == 3
            assert label.pauli_at(2) == SinglePauli.Y
            assert label.pauli_at(6) == SinglePauli.Z
            assert label.pauli_at(4) == SinglePauli.X

    def test_construct_from_str_with_spaces(self) -> None:
        """Construct from a string with redundant spaces: ' Y 2  Z 6  X 4 '."""
        for label in (
            PauliLabel.from_str(" Y 2  Z 6  X 4 "),
            pauli_label(" Y 2  Z 6  X 4 "),
        ):
            assert len(label) == 3
            assert label.pauli_at(2) == SinglePauli.Y
            assert label.pauli_at(6) == SinglePauli.Z
            assert label.pauli_at(4) == SinglePauli.X

    @pytest.mark.parametrize(
        "label_str",
        [
            "Y2 Z6 A4",
            "Y2 Z6 YZ4",
            "Y2 Z6 3 X4",
        ],
    )
    def test_construct_from_invalid_pauli_label_str_fails(self, label_str: str) -> None:
        with pytest.raises(ValueError):
            pauli_label(label_str)

    def test_construct_from_duplicate_paulis_fails(self) -> None:
        with pytest.raises(ValueError):
            pauli_label("Y2 Z6 X2")

    def test_construct_with_index_and_pauli_list(self) -> None:
        label = PauliLabel.from_index_and_pauli_list(
            index=[2, 6, 4], pauli=[SinglePauli.Y, SinglePauli.Z, SinglePauli.X]
        )
        assert len(label) == 3
        assert label.pauli_at(2) == SinglePauli.Y
        assert label.pauli_at(6) == SinglePauli.Z
        assert label.pauli_at(4) == SinglePauli.X

    def test_construct_with_index_and_pauli_list_fails_when_list_lengths_unmatch(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            PauliLabel.from_index_and_pauli_list(
                index=[2, 6], pauli=[SinglePauli.Y, SinglePauli.Z, SinglePauli.X]
            )

        with pytest.raises(ValueError):
            PauliLabel.from_index_and_pauli_list(
                index=[2, 6, 4], pauli=[SinglePauli.Y, SinglePauli.Z]
            )

    def test_construct_with_pauli_label_provider(self) -> None:
        class Provider:
            def get_index_list(self) -> list[int]:
                return [2, 6, 4]

            def get_pauli_id_list(self) -> list[int]:
                return [SinglePauli.Y, SinglePauli.Z, SinglePauli.X]

        p = Provider()
        for label in (
            PauliLabel.of(p),
            pauli_label(p),
        ):
            assert len(label) == 3
            assert label.pauli_at(2) == SinglePauli.Y
            assert label.pauli_at(6) == SinglePauli.Z
            assert label.pauli_at(4) == SinglePauli.X

    def test_qubit_indices_in(self) -> None:
        label = pauli_label(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))
        )
        for i in [2, 6, 4]:
            assert i in label.qubit_indices()

    def test_qubit_indices_iterator(self) -> None:
        label = pauli_label(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))
        )
        expected = [2, 6, 4]
        for i in label.qubit_indices():
            assert i in expected

    def test_string_representation(self) -> None:
        label = pauli_label(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))
        )
        assert str(label) == "Y2 X4 Z6"

    def test_equal_hash(self) -> None:
        label1 = pauli_label(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))
        )
        label2 = pauli_label(
            ((6, SinglePauli.Z), (2, SinglePauli.Y), (4, SinglePauli.X))
        )
        assert label1 == label2
        assert hash(label1) == hash(label2)

        label3 = pauli_label(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (3, SinglePauli.X))
        )
        assert label1 != label3
        assert hash(label1) != hash(label3)

        label4 = pauli_label(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.Z))
        )
        assert label1 != label4
        assert hash(label1) != hash(label4)

    def test_index_and_pauli_id_list(self) -> None:
        label = pauli_label(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))
        )
        index_list, pauli_id_list = label.index_and_pauli_id_list
        assert set(zip(index_list, pauli_id_list)) == set(
            ((2, SinglePauli.Y), (6, SinglePauli.Z), (4, SinglePauli.X))
        )

    def test_pauli_cache(self) -> None:
        with patch(  # type: ignore
            "quri_parts.core.operator.pauli._pauli_cache", WeakValueDictionary()
        ) as pauli_cache:
            # Checks if the cache works correctly.
            pl_1 = pauli_label("X0 X1 Y2 Y3 Z4 Z5")
            pl_2 = pauli_label("X0 X1 Y2 Y3 Z4 Z5")
            assert id(pl_1) == id(pl_2)

            cache_len = len(pauli_cache)
            pl_2 = pauli_label("X0 X1 Y2 Y3 Z4 Z7")
            assert id(pl_1) != id(pl_2)
            assert len(pauli_cache) == cache_len + 1

            pl_3 = pauli_label("X0 X1 Y2 Y3 Z4 Z6")
            assert id(pl_1) != id(pl_3)

            # Checks if the cache is cleared correctly.
            cache_len = len(pauli_cache)
            pl_3 = pauli_label("X0 X1 Y2 Y3 Z4 Z8")
            assert str(pl_3) == "X0 X1 Y2 Y3 Z4 Z8"
            assert id(pl_1) != id(pl_3)
            assert len(pauli_cache) == cache_len

            pl_3 = "QURI Parts"  # type: ignore
            assert len(pauli_cache) == cache_len - 1


class TestPauliProduct:
    def test_pauli_product(self) -> None:
        pauli1 = pauli_label("X0 Y2 Z4 Z5")
        pauli2 = pauli_label("Y0 Y2 Y4 X5 Z6")

        prod = pauli_product(pauli1, pauli2)

        assert prod == (pauli_label("Z0 X4 Y5 Z6"), 1.0j)

    def test_pauli_multiplied_by_identitiy(self) -> None:
        pauli1 = pauli_label("X0 Y2 Z4 Z5")

        prod_right = pauli_product(pauli1, PAULI_IDENTITY)
        prod_left = pauli_product(PAULI_IDENTITY, pauli1)

        assert prod_right == (pauli1, 1.0)
        assert prod_left == (pauli1, 1.0)
