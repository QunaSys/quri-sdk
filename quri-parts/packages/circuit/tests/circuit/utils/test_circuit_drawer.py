# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Any

import pytest

from quri_parts.circuit import (
    CNOT,
    RZ,
    SWAP,
    ParametricQuantumCircuit,
    PauliRotation,
    QuantumCircuit,
    UnitaryMatrix,
    X,
)
from quri_parts.circuit.utils.circuit_drawer import (
    _generate_gate_aa,
    circuit_to_string,
    draw_circuit,
)


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
        expected = ["  ___  ", " | X | ", "-|999|-", " |___| "]
        assert _generate_gate_aa(X(0), gate_idx=1000) == expected


def test_circuit_to_string_many_gates() -> None:
    # Regression test: a gate index beyond 999 used to widen the gate cell
    # past its fixed width and raise a ValueError when rendered.
    circuit = QuantumCircuit(1)
    for _ in range(1001):
        circuit.add_X_gate(0)

    with pytest.warns(Warning):
        result = circuit_to_string(circuit)
    assert result

    # __repr__ skips the (very wide) full ASCII drawing for circuits this
    # large and returns a compact representation instead, without warning.
    assert repr(circuit) == "<QuantumCircuit qubit_count=1 gate_count=1001>"


def test_circuit_to_string_zero_qubits() -> None:
    # Regression test: a 0-qubit circuit used to render as "", indistinguishable
    # from an empty list when nested, e.g. repr([circuit]) == "[]".
    circuit = QuantumCircuit(0)
    assert circuit_to_string(circuit) == "<QuantumCircuit qubit_count=0>"
    assert repr(circuit) == "<QuantumCircuit qubit_count=0>"


def test_repr_falls_back_when_drawer_unavailable() -> None:
    # Regression test: quri-parts-rust can be used without the quri-parts-circuit
    # package installed, in which case the ASCII-art drawer module is missing.
    # __repr__ must not raise in that case.
    circuit = QuantumCircuit(2)
    circuit.add_X_gate(0)

    parametric_circuit = ParametricQuantumCircuit(2)
    parametric_circuit.add_ParametricRX_gate(0)

    module_name = "quri_parts.circuit.utils.circuit_drawer"
    original = sys.modules.pop(module_name)
    sys.modules[module_name] = None  # type: ignore[assignment]
    try:
        assert repr(circuit) == "<QuantumCircuit qubit_count=2 gate_count=1>"
        assert (
            repr(parametric_circuit)
            == "<ParametricQuantumCircuit qubit_count=2 gate_count=1>"
        )
    finally:
        del sys.modules[module_name]
        sys.modules[module_name] = original
