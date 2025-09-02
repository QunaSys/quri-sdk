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

from quri_parts.core.operator import (
    PAULI_IDENTITY,
    CommutablePauliSet,
    Operator,
    PauliLabel,
    pauli_label,
)
from quri_parts.core.operator.grouping import (
    individual_pauli_grouping,
    sorted_injection_grouping,
)
from quri_parts.core.operator.grouping.pauli_grouping import bitwise_pauli_grouping


class TestIndividualPauliGrouping:
    def test_grouping_pauli_labels(self) -> None:
        paulis = [
            pauli_label("Z0"),
            pauli_label("Z0"),
            pauli_label("X1 Y2"),
            pauli_label("X1 Z3"),
            PAULI_IDENTITY,
            pauli_label("X2 Z4"),
            PAULI_IDENTITY,
        ]

        groups = individual_pauli_grouping(paulis)
        assert groups == frozenset(
            {
                frozenset({PAULI_IDENTITY}),
                frozenset({pauli_label("Z0")}),
                frozenset({pauli_label("X1 Y2")}),
                frozenset({pauli_label("X1 Z3")}),
                frozenset({pauli_label("X2 Z4")}),
            }
        )

    def test_grouping_operator(self) -> None:
        op = Operator(
            {
                pauli_label("Z0"): 0.1,
                pauli_label("X1 Y2"): 0.2,
                pauli_label("X1 Z3"): 0.3,
                PAULI_IDENTITY: 0.4,
                pauli_label("X2 Z4"): 0.5,
            }
        )

        groups = individual_pauli_grouping(op)
        assert groups == frozenset(
            {
                frozenset({PAULI_IDENTITY}),
                frozenset({pauli_label("Z0")}),
                frozenset({pauli_label("X1 Y2")}),
                frozenset({pauli_label("X1 Z3")}),
                frozenset({pauli_label("X2 Z4")}),
            }
        )


class TestBitwisePauliGrouping:
    def test_only_all_XYZ_grouping(self) -> None:
        paulis = [
            pauli_label("X0 X1"),
            pauli_label("X1 X2"),
            PAULI_IDENTITY,
            pauli_label("Y3 Y4"),
            pauli_label("Y4 Y6"),
            pauli_label("Z0 Z3"),
            pauli_label("Z1 Z4 Z6"),
        ]

        groups = bitwise_pauli_grouping(paulis)
        assert groups == frozenset(
            {
                frozenset({PAULI_IDENTITY}),
                frozenset({pauli_label("X0 X1"), pauli_label("X1 X2")}),
                frozenset({pauli_label("Y3 Y4"), pauli_label("Y4 Y6")}),
                frozenset({pauli_label("Z0 Z3"), pauli_label("Z1 Z4 Z6")}),
            }
        )

    def test_bitwise_grouping(self) -> None:
        paulis = [
            pauli_label("X0 X1"),  # group X
            pauli_label("X1 Y2"),  # group 1
            pauli_label("Y2 Z3"),  # group 1
            PAULI_IDENTITY,
            pauli_label("Y2 Z3 X4"),  # group 1
            pauli_label("Y1 Y2"),  # group Y
            pauli_label("Y2 Z4 X5"),  # group 2
            pauli_label("X0"),  # group X
            pauli_label("Y4 X6"),  # group 3
            pauli_label("Z2 Z3"),  # group Z
            pauli_label("X3 Y5 Z7"),  # group 3
            pauli_label("Y1 Y3"),  # group Y
            pauli_label("Z5"),  # group Z
        ]
        op = Operator({p: 0.1 * (i + 1) for i, p in enumerate(paulis)})

        for input in (paulis, op):
            groups = bitwise_pauli_grouping(input)
            assert groups == frozenset(
                {
                    frozenset({PAULI_IDENTITY}),
                    # group X
                    frozenset(
                        {
                            pauli_label("X0 X1"),
                            pauli_label("X0"),
                        }
                    ),
                    # group Y
                    frozenset(
                        {
                            pauli_label("Y1 Y3"),
                            pauli_label("Y1 Y2"),
                        }
                    ),
                    # group Z
                    frozenset({pauli_label("Z2 Z3"), pauli_label("Z5")}),
                    # group 1
                    frozenset(
                        {
                            pauli_label("X1 Y2"),
                            pauli_label("Y2 Z3"),
                            pauli_label("Y2 Z3 X4"),
                        }
                    ),
                    # group 2
                    frozenset(
                        {
                            pauli_label("Y2 Z4 X5"),
                        }
                    ),
                    # group 3
                    frozenset(
                        {
                            pauli_label("Y4 X6"),
                            pauli_label("X3 Y5 Z7"),
                        }
                    ),
                }
            )


class TestSortedInjectionGrouping:
    @pytest.mark.parametrize(
        "op_dict, expected_groups",
        [
            (
                {
                    pauli_label("Z1 Z4 Z6"): 0.5,
                    pauli_label("Y4 Y6"): 0.4,
                    pauli_label("Z0 Z3"): 0.3,
                    PAULI_IDENTITY: 0.2,
                    pauli_label("X0 X1"): 0.1,
                },
                frozenset(
                    [
                        frozenset(
                            {
                                pauli_label("Z1 Z4 Z6"),
                                pauli_label("Z0 Z3"),
                                PAULI_IDENTITY,
                            }
                        ),
                        frozenset(
                            {
                                pauli_label("Y4 Y6"),
                                pauli_label("X0 X1"),
                            }
                        ),
                    ]
                ),
            ),
            (
                {
                    pauli_label("Z1 Z4 Z6"): -0.5,
                    pauli_label("Y4 Y6"): -0.4,
                    pauli_label("Z0 Z3"): 0.3,
                    PAULI_IDENTITY: 0.2,
                    pauli_label("X0 X1"): -0.1,
                },
                frozenset(
                    [
                        frozenset(
                            {
                                pauli_label("Z1 Z4 Z6"),
                                pauli_label("Z0 Z3"),
                                PAULI_IDENTITY,
                            }
                        ),
                        frozenset(
                            {
                                pauli_label("Y4 Y6"),
                                pauli_label("X0 X1"),
                            }
                        ),
                    ]
                ),
            ),
            (
                {
                    PAULI_IDENTITY: 0.7,
                    pauli_label("Z1 Z4 Z6"): -0.5,
                    pauli_label("Y4 Y6"): -0.4,
                    pauli_label("X0 X1"): -0.3,
                    pauli_label("Z0 Z3"): 0.1,
                },
                frozenset(
                    [
                        frozenset(
                            {
                                PAULI_IDENTITY,
                                pauli_label("Z1 Z4 Z6"),
                                pauli_label("Z0 Z3"),
                            }
                        ),
                        frozenset(
                            {
                                pauli_label("Y4 Y6"),
                                pauli_label("X0 X1"),
                            }
                        ),
                    ]
                ),
            ),
        ],
    )
    def test_non_abs_grouping(
        self,
        op_dict: dict[PauliLabel, complex],
        expected_groups: frozenset[CommutablePauliSet],
    ) -> None:
        op = Operator(op_dict)

        groups = frozenset(sorted_injection_grouping(op))
        assert groups == expected_groups
