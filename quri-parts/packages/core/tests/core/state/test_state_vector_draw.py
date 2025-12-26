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

from quri_parts.circuit import ParametricQuantumCircuit, QuantumCircuit
from quri_parts.core.state import ParametricQuantumStateVector, QuantumStateVector


def test_draw_latex_superposition() -> None:
    amp = 1 / np.sqrt(2)
    state = QuantumStateVector(2, vector=[amp, 0, 0, amp])

    rendered = state.draw(output="latex_source", precision=6)

    assert (
        rendered == "\\frac{\\sqrt{2}}{2}|00\\rangle + \\frac{\\sqrt{2}}{2}|11\\rangle"
    )


def test_draw_text_truncation() -> None:
    state = QuantumStateVector(2, vector=[0.5, 0.5, 0.5, 0.5])

    rendered = state.draw(output="text", max_terms=2, precision=3)

    assert "|00>" in rendered
    assert "|01>" in rendered or "|10>" in rendered or "|11>" in rendered
    assert "terms truncated" in rendered


def test_parametric_state_draw() -> None:
    circuit = ParametricQuantumCircuit(1)
    state = ParametricQuantumStateVector(1, circuit, [0, 1])

    rendered = state.draw(output="latex_source", precision=3)

    assert rendered == "1|1\\rangle"


def test_draw_latex_common_pi_fraction() -> None:
    state = QuantumStateVector(1, vector=[0, 1j * np.pi / 4])

    rendered = state.draw(output="latex_source", precision=6)

    assert rendered == "\\frac{\\pi}{4}i|1\\rangle"


def test_draw_warns_when_circuit_attached() -> None:
    circuit = QuantumCircuit(1)
    circuit.add_X_gate(0)  # ensure a gate exists
    state = QuantumStateVector(1, circuit=circuit.freeze())

    with pytest.warns(UserWarning, match="ignores any attached circuit"):
        rendered = state.draw(output="text")
    assert "|0" in rendered or "|1" in rendered
