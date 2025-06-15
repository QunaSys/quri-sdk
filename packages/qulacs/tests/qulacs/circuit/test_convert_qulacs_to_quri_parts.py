# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy import cos, exp, pi, sin
from qulacs import QuantumCircuit as QulacsQuantumCircuit
from qulacs.gate import (
    CNOT,
    CZ,
    P0,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    Identity,
    Pauli,
    PauliRotation,
)

from quri_parts.circuit import QuantumCircuit, gates
from quri_parts.qulacs.circuit import circuit_from_qulacs

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


def Rz_mat(theta: float) -> "npt.NDArray[np.complex128]":
    return np.array([[exp(1j * theta / 2), 0], [0, exp(-1j * theta / 2)]])


def Rx_mat(theta: float) -> "npt.NDArray[np.complex128]":
    return np.array(
        [[cos(theta / 2), 1j * sin(theta / 2)], [1j * sin(theta / 2), cos(theta / 2)]]
    )


def test_circuit_from_qulacs() -> None:
    qul_circ = QulacsQuantumCircuit(10)
    qul_circ.add_gate(Identity(0))
    qul_circ.add_X_gate(1)
    qul_circ.add_Y_gate(2)
    qul_circ.add_Z_gate(3)
    qul_circ.add_H_gate(4)
    qul_circ.add_S_gate(5)
    qul_circ.add_Sdag_gate(6)
    qul_circ.add_T_gate(0)
    qul_circ.add_Tdag_gate(1)
    qul_circ.add_sqrtX_gate(2)
    qul_circ.add_sqrtXdag_gate(3)
    qul_circ.add_sqrtY_gate(4)
    qul_circ.add_sqrtYdag_gate(5)
    qul_circ.add_RX_gate(6, -0.125)
    qul_circ.add_RY_gate(0, -0.250)
    qul_circ.add_RZ_gate(1, -0.500)
    qul_circ.add_gate(CNOT(2, 3))
    qul_circ.add_gate(CZ(4, 5))
    qul_circ.add_gate(SWAP(0, 1))
    qul_circ.add_gate(TOFFOLI(7, 8, 9))
    qul_circ.add_gate(U1(0, pi))
    qul_circ.add_gate(U2(1, pi, pi))
    qul_circ.add_gate(U3(2, pi, pi, pi))
    qul_circ.add_gate(Pauli([0, 3, 5], [1, 3, 1]))
    qul_circ.add_gate(PauliRotation([6, 8, 9], [2, 3, 1], -0.5))

    U1_mat = -1j * Rz_mat(pi)
    U2_mat = -1 * Rz_mat(3 * pi / 2) @ Rx_mat(pi / 2) @ Rz_mat(pi / 2)
    U3_mat = (
        -1
        * Rz_mat(4 * pi)
        @ Rx_mat(pi / 2)
        @ Rz_mat(2 * pi)
        @ Rx_mat(pi / 2)
        @ Rz_mat(pi)
    )

    gate_list = [
        gates.Identity(0),
        gates.X(1),
        gates.Y(2),
        gates.Z(3),
        gates.H(4),
        gates.S(5),
        gates.Sdag(6),
        gates.T(0),
        gates.Tdag(1),
        gates.SqrtX(2),
        gates.SqrtXdag(3),
        gates.SqrtY(4),
        gates.SqrtYdag(5),
        gates.RX(6, np.round(0.125, 5)),
        gates.RY(0, np.round(0.250, 5)),
        gates.RZ(1, np.round(0.500, 5)),
        gates.CNOT(2, 3),
        gates.CZ(4, 5),
        gates.SWAP(0, 1),
        gates.TOFFOLI(7, 8, 9),
        gates.UnitaryMatrix([0], np.round(U1_mat, 5)),
        gates.UnitaryMatrix([1], np.round(U2_mat, 5)),
        gates.UnitaryMatrix([2], np.round(U3_mat, 5)),
        gates.Pauli([0, 3, 5], [1, 3, 1]),
        gates.PauliRotation([6, 8, 9], [2, 3, 1], 0.5),
    ]
    expected = QuantumCircuit(10, gates=gate_list)

    assert circuit_from_qulacs(qul_circ).gates == expected.gates

    # test for empty circuit
    emp_circ = QulacsQuantumCircuit(10)
    assert circuit_from_qulacs(emp_circ) == QuantumCircuit(10)

    # test for error circuit
    err_circ = QulacsQuantumCircuit(10)
    err_circ.add_gate(P0(0))
    with pytest.raises(ValueError):
        circuit_from_qulacs(err_circ)
