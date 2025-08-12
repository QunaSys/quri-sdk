# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import ParametricQuantumCircuit
from quri_parts.core.estimator import Estimatable
from quri_parts.core.estimator.utils import is_estimatable
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label, zero
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
)


def test_is_estimatable() -> None:
    states = [
        GeneralCircuitQuantumState(3),
        ComputationalBasisState(3),
        QuantumStateVector(3),
        ParametricCircuitQuantumState(3, circuit=ParametricQuantumCircuit(3)),
        ParametricQuantumStateVector(3, circuit=ParametricQuantumCircuit(3)),
    ]

    valid_pauli_labels: list[Estimatable] = [
        zero(),
        PAULI_IDENTITY,
        pauli_label("X0"),
        pauli_label("Y1"),
        pauli_label("Z2"),
        pauli_label("X0 Y2"),
        pauli_label("Y1 Z2"),
        pauli_label("Z2"),
    ]

    for op in valid_pauli_labels:
        for state in states:
            assert is_estimatable(op, state)

    invalid_pauli_labels = [pauli_label("X3"), pauli_label("Z50000")]
    for op in invalid_pauli_labels:
        for state in states:
            assert not is_estimatable(op, state)

    valid_operator = Operator(
        {
            pauli_label("Z0 Z1 Z2"): 1,
            pauli_label("X0 Z1 Y2"): 2,
            pauli_label("X0 Z1 X2"): 2,
            pauli_label("Z1"): 2,
            PAULI_IDENTITY: 3,
        }
    )

    for state in states:
        assert is_estimatable(valid_operator, state)

    invalid_operator = Operator(
        {
            pauli_label("Z50000"): 1,
            pauli_label("Z3"): 1,
            pauli_label("Z0 Z1 Z2"): 1,
            pauli_label("X0 Z1 Y2"): 2,
            pauli_label("X0 Z1 X2"): 2,
            pauli_label("Z1"): 2,
            PAULI_IDENTITY: 3,
        }
    )
    for state in states:
        assert not is_estimatable(invalid_operator, state)
