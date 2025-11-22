# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import PauliRotation
from quri_parts.core.circuit.exp_single_pauli_gate import convert_exp_single_pauli_gate
from quri_parts.core.operator import pauli_label


def test_convert_exp_single_pauli() -> None:
    test_pauli = pauli_label("Z0 X1 Y2 Z3")
    test_coef = 1.7
    ts_gates = convert_exp_single_pauli_gate(test_pauli, test_coef)

    expected_gate = PauliRotation((1, 0, 3, 2), (1, 3, 3, 2), -3.4)

    assert ts_gates == expected_gate
