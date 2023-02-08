# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cmath
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from quri_parts.circuit import QuantumGate, gate_names, gates

from .transpiler import GateDecomposer


def su2_decompose(
    ut: Sequence[Sequence[complex]], eps: float = 1e-15
) -> npt.NDArray[np.float64]:

    if abs(ut[0][1]) < eps or abs(ut[1][0]) < eps:
        theta = np.zeros(4)
        theta[0] = (1j * cmath.log(ut[0][0] * ut[1][1])).real
        theta[1] = (1j * cmath.log(ut[0][0] / ut[1][1])).real

        m = np.zeros(4)
        a = (cmath.phase(ut[0][0]) + (theta[0] + theta[1]) / 2) / (-np.pi / 2)
        b = (cmath.phase(ut[1][1]) + (theta[0] - theta[1]) / 2) / (-np.pi / 2)
        m[0] = (a + b) / 2
        m[1] = (a - b) / 2
        theta += np.pi * m

        return theta

    elif abs(ut[0][0]) < eps or abs(ut[1][1]) < eps:
        theta = np.zeros(4)
        theta[0] = (1j * cmath.log(-ut[1][0] * ut[0][1])).real
        theta[1] = (1j * cmath.log(-ut[1][0] / ut[0][1])).real
        theta[2] = np.pi

        m = np.zeros(4)
        a = (cmath.phase(ut[1][0]) + (theta[0] + theta[1]) / 2) / (-np.pi / 2)
        b = (cmath.phase(-ut[0][1]) + (theta[0] - theta[1]) / 2) / (-np.pi / 2)
        m[0] = (a + b) / 2
        m[1] = (a - b) / 2
        theta += np.pi * m

        return theta

    else:
        theta = np.zeros(4)
        theta[0] = (1j * cmath.log(ut[0][0] * ut[1][1] - ut[1][0] * ut[0][1])).real
        theta[1] = (0.5j * cmath.log(-ut[0][0] * ut[1][0] / (ut[1][1] * ut[0][1]))).real
        if abs(ut[0][0] * ut[1][1] + ut[1][0] * ut[0][1]) <= 1:
            theta[2] = math.acos(abs(ut[0][0] * ut[1][1] + ut[1][0] * ut[0][1]))
        theta[3] = (0.5j * cmath.log(-ut[0][0] * ut[0][1] / (ut[1][1] * ut[1][0]))).real

        m = np.zeros(4)
        if abs(abs(ut[0][0]) - math.cos(theta[2] / 2)) > abs(
            abs(ut[0][0]) - math.sin(theta[2] / 2)
        ):
            theta[2] = np.pi - theta[2]
        a = (cmath.phase(ut[0][0]) + (theta[0] + theta[3] + theta[1]) / 2) / (
            -np.pi / 4
        )
        b = (cmath.phase(ut[1][0]) + (theta[0] - theta[3] + theta[1]) / 2) / (
            -np.pi / 4
        )
        c = (cmath.phase(-ut[0][1]) + (theta[0] + theta[3] - theta[1]) / 2) / (
            -np.pi / 4
        )
        m[0] = (b + c) / 2
        m[1] = (a - c) / 2
        m[3] = (a - b) / 2
        theta += np.pi / 2 * m

        return theta


class SingleQubitUnitaryMatrix2RYRZTranspiler(GateDecomposer):
    """CircuitTranspiler, which decomposes single qubit UnitaryMatrix gates
    into gate sequences containing RY and RZ gates.

    Ref:
        [1]: Tomonori Shirakawa, Hiroshi Ueda, and Seiji Yunoki,
            Automatic quantum circuit encoding of a given arbitrary quantum state,
            arXiv:2112.14524v1, p.22, (2021).
    """

    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.name == gate_names.UnitaryMatrix and len(gate.target_indices) == 1

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        theta = su2_decompose(gate.unitary_matrix)

        target = gate.target_indices[0]
        return [
            gates.RZ(target, theta[1]),
            gates.RY(target, theta[2]),
            gates.RZ(target, theta[3]),
        ]
