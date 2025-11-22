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
from stim import Circuit as StimCircuit

from quri_parts.circuit import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    U1,
    U2,
    U3,
    H,
    Pauli,
    PauliRotation,
    QuantumCircuit,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    X,
    Y,
    Z,
)
from quri_parts.stim.circuit import convert_circuit, convert_gate


def test_convert_gate() -> None:
    assert convert_gate(X(0)) == [("X", [0])]
    assert convert_gate(Y(1)) == [("Y", [1])]
    assert convert_gate(Z(2)) == [("Z", [2])]
    assert convert_gate(H(3)) == [("H", [3])]
    assert convert_gate(S(4)) == [("S", [4])]
    assert convert_gate(Sdag(5)) == [("S_DAG", [5])]
    assert convert_gate(SqrtX(6)) == [("SQRT_X", [6])]
    assert convert_gate(SqrtXdag(7)) == [("SQRT_X_DAG", [7])]
    assert convert_gate(SqrtY(8)) == [("SQRT_Y", [8])]
    assert convert_gate(SqrtYdag(9)) == [("SQRT_Y_DAG", [9])]

    with pytest.raises(ValueError):
        convert_gate(RX(0, 1.0))
    assert convert_gate(RX(0, np.pi / 2)) == [("SQRT_X", [0])]

    with pytest.raises(ValueError):
        convert_gate(RY(0, 2.0))
    assert convert_gate(RY(0, np.pi)) == [("Y", [0])]

    with pytest.raises(ValueError):
        convert_gate(RZ(0, 3.0))
    assert convert_gate(RZ(0, 3 * np.pi / 2)) == [("S_DAG", [0])]

    with pytest.raises(ValueError):
        convert_gate(U1(0, 1.0))
    assert convert_gate(U1(0, np.pi / 2)) == [("S", [0])]

    with pytest.raises(ValueError):
        convert_gate(U2(0, 1.0, np.pi))
    assert convert_gate(U2(0, np.pi, np.pi / 2)) == [
        ("I", [0]),
        ("SQRT_X", [0]),
        ("S_DAG", [0]),
    ]

    with pytest.raises(ValueError):
        convert_gate(U3(0, 1.0, 2.0, np.pi))
    assert convert_gate(U3(0, np.pi, -np.pi / 2, 3 * np.pi / 2)) == [
        ("S_DAG", [0]),
        ("SQRT_X", [0]),
        ("I", [0]),
        ("SQRT_X", [0]),
        ("S", [0]),
    ]

    assert convert_gate(CNOT(0, 1)) == [("CNOT", [0, 1])]
    assert convert_gate(CZ(0, 1)) == [("CZ", [0, 1])]
    assert convert_gate(SWAP(0, 1)) == [("SWAP", [0, 1])]
    assert convert_gate(Pauli([0, 1, 2], [1, 2, 3])) == [
        ("X", [0]),
        ("Y", [1]),
        ("Z", [2]),
    ]
    with pytest.raises(ValueError):
        convert_gate(PauliRotation([0, 1], [1, 2], 1.0))
    assert convert_gate(PauliRotation([0, 2, 4], [1, 2, 3], np.pi / 2)) == [
        ("H", [0]),
        ("SQRT_X", [2]),
        ("CNOT", [4, 0]),
        ("CNOT", [2, 0]),
        ("S", [0]),
        ("CNOT", [2, 0]),
        ("CNOT", [4, 0]),
        ("H", [0]),
        ("SQRT_X_DAG", [2]),
    ]


def test_convert_circuit() -> None:
    qubit_count = 5

    qp_circ = QuantumCircuit(qubit_count)
    assert convert_circuit(qp_circ) == StimCircuit()

    qp_circ.add_H_gate(1)
    qp_circ.add_CZ_gate(1, 0)
    qp_circ.add_RX_gate(2, np.pi / 2)
    qp_circ.add_PauliRotation_gate([1, 3, 4], [1, 2, 3], 3 * np.pi / 2)
    qp_circ.add_U3_gate(4, np.pi, 3 * np.pi / 2, -np.pi / 2)

    stim_circuit = StimCircuit(
        """
        H 1
        CZ 1 0
        SQRT_X 2
        H 1
        SQRT_X 3
        CNOT 4 1 3 1
        S_DAG 1
        CNOT 3 1 4 1
        H 1
        SQRT_X_DAG 3
        S_DAG 4
        SQRT_X 4
        I 4
        SQRT_X 4
        S 4
    """
    )
    assert convert_circuit(qp_circ) == stim_circuit
