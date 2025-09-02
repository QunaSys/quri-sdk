# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import pytest

from quri_parts.circuit import (
    CNOT,
    RZ,
    SWAP,
    PauliRotation,
    QuantumCircuit,
    UnitaryMatrix,
    X,
)
from quri_parts.circuit.utils.circuit_drawer import _generate_gate_aa, draw_circuit


def test_draw_empty_circuit(capsys: pytest.CaptureFixture[Any]) -> None:
    draw_circuit(QuantumCircuit(1))
    expected = " \n \n-\n \n"
    assert capsys.readouterr().out == expected

    draw_circuit(QuantumCircuit(2))
    expected = " \n \n-\n \n \n \n-\n \n"
    assert capsys.readouterr().out == expected


def test_generate_gate_aa() -> None:
    expected = ["  ___  ", " | X | ", "-|0  |-", " |___| "]
    assert _generate_gate_aa(X(0), gate_idx=0) == expected

    expected = ["  ___  ", " |RZ | ", "-|2  |-", " |___| "]
    assert _generate_gate_aa(RZ(0, 0.1), gate_idx=2) == expected

    expected = [
        "  ___  ",
        " |CX | ",
        "-|5  |-",
        " |___| ",
        "   |   ",
        "   |   ",
        "   ●   ",
    ]
    assert _generate_gate_aa(CNOT(1, 0), gate_idx=5) == expected

    expected = [
        "       ",
        "  4    ",
        "---x---",
        "   |   ",
        "   |   ",
        "   |   ",
        "---|---",
        "   |   ",
        "   |   ",
        "   |   ",
        "---x---",
        "       ",
    ]
    assert _generate_gate_aa(SWAP(0, 2), gate_idx=4) == expected

    expected = [
        "  ___  ",
        " |PR | ",
        "-|5  |-",
        " |_ _| ",
        "  | |  ",
        "  | |  ",
        "--| |--",
        "  | |  ",
        " _| |_ ",
        " |   | ",
        "-|   |-",
        " |___| ",
    ]
    assert _generate_gate_aa(PauliRotation((0, 2), (0, 1), 0.1), gate_idx=5) == expected

    expected = [
        "  ___  ",
        " |Mat| ",
        "-|6  |-",
        " |   | ",
        " |   | ",
        " |   | ",
        "-|   |-",
        " |___| ",
    ]
    assert (
        _generate_gate_aa(
            UnitaryMatrix(
                target_indices=(0, 1),
                unitary_matrix=[
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            ),
            gate_idx=6,
        )
        == expected
    )

    with pytest.warns(Warning):
        _generate_gate_aa(X(0), gate_idx=1000)
