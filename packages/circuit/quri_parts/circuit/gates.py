# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Literal

import numpy as np

from quri_parts.circuit import gate_names

from .gate import ParametricQuantumGate, QuantumGate


class IdentityFactory:
    name: Literal["Identity"] = gate_names.Identity

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


Identity = IdentityFactory()
r"""Identity gate represented by matrix
:math:`\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}`"""


class XFactory:
    name: Literal["X"] = gate_names.X

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


X = XFactory()
r"""Pauli X gate represented by matrix
:math:`\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`"""


class YFactory:
    name: Literal["Y"] = gate_names.Y

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


Y = YFactory()
r"""Pauli Y gate represented by matrix
:math:`\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}`"""


class ZFactory:
    name: Literal["Z"] = gate_names.Z

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


Z = ZFactory()
r"""Pauli Z gate represented by matrix
:math:`\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}`"""


class HFactory:
    name: Literal["H"] = gate_names.H

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


H = HFactory()
r"""Hadamard gate represented by matrix
:math:`\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}`"""


class SFactory:
    name: Literal["S"] = gate_names.S

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


S = SFactory()
r"""S Gate, or sqrt(Z) gate represented by matrix
:math:`\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}`"""


class SdagFactory:
    name: Literal["Sdag"] = gate_names.Sdag

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


Sdag = SdagFactory()
r"""Sdag Gate, conjugate of S gate represented by matrix
:math:`\begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}`"""


class SqrtXFactory:
    name: Literal["SqrtX"] = gate_names.SqrtX

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


SqrtX = SqrtXFactory()
r"""Square root of X gate represented by matrix
:math:`\frac{1}{2}\begin{pmatrix} 1+i & 1-i \\ 1-i & 1+i \end{pmatrix}`
"""


class SqrtXdagFactory:
    name: Literal["SqrtXdag"] = gate_names.SqrtXdag

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


SqrtXdag = SqrtXdagFactory()
r"""Conjugate of the quare root of X gate represented by matrix
:math:`\frac{1}{2}\begin{pmatrix} 1-i & 1+i \\ 1+i & 1-i \end{pmatrix}`
"""


class SqrtYFactory:
    name: Literal["SqrtY"] = gate_names.SqrtY

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


SqrtY = SqrtYFactory()
r"""Square root of Y gate represented by matrix
:math:`\frac{1+i}{2}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}`
"""


class SqrtYdagFactory:
    name: Literal["SqrtYdag"] = gate_names.SqrtYdag

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


SqrtYdag = SqrtYdagFactory()
r"""Conjugate of the quare root of Y gate represented by matrix
:math:`\frac{1-i}{2}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}`
"""


class TFactory:
    name: Literal["T"] = gate_names.T

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


T = TFactory()
r"""T gate, or sqrt(S) gate represented by matrix
:math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}`
"""


class TdagFactory:
    name: Literal["Tdag"] = gate_names.Tdag

    def __call__(self, target_index: int) -> QuantumGate:
        return QuantumGate(name=self.name, target_indices=(target_index,))


Tdag = TdagFactory()
r"""Conjugate of the T gate represented by matrix
:math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}`
"""


class RXFactory:
    name: Literal["RX"] = gate_names.RX

    def __call__(self, target_index: int, angle: float) -> QuantumGate:
        return QuantumGate(
            name=self.name, target_indices=(target_index,), params=(angle,)
        )


RX = RXFactory()
r"""RX gate equivalant to :math:`\exp(-i\theta X/2)` represented by matrix
:math:`\begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
-i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}`
"""


class RYFactory:
    name: Literal["RY"] = gate_names.RY

    def __call__(self, target_index: int, angle: float) -> QuantumGate:
        return QuantumGate(
            name=self.name, target_indices=(target_index,), params=(angle,)
        )


RY = RYFactory()
r"""RY gate equivalant to :math:`\exp(-i\theta Y/2)` represented by matrix
:math:`\begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}`
"""


class RZFactory:
    name: Literal["RZ"] = gate_names.RZ

    def __call__(self, target_index: int, angle: float) -> QuantumGate:
        return QuantumGate(
            name=self.name, target_indices=(target_index,), params=(angle,)
        )


RZ = RZFactory()
r"""RZ gate equivalant to :math:`\exp(-i\theta Z/2)` represented by matrix
:math:`\begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}`
"""


class U1Factory:
    name: Literal["U1"] = gate_names.U1

    def __call__(self, target_index: int, lmd: float) -> QuantumGate:
        return QuantumGate(
            name=self.name, target_indices=(target_index,), params=(lmd,)
        )


U1 = U1Factory()
r"""U1 gate is a single-qubit rotation about the Z axis:

:math:`U_1(\lambda) = e^{i\lambda/2} R_Z(\lambda)`.
Represented by matrix :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{pmatrix}`
"""


class U2Factory:
    name: Literal["U2"] = gate_names.U2

    def __call__(self, target_index: int, phi: float, lmd: float) -> QuantumGate:
        return QuantumGate(
            name=self.name, target_indices=(target_index,), params=(phi, lmd)
        )


U2 = U2Factory()
r"""U2 gate is a single-qubit rotation about X + Z axis:

:math:`U_2(\phi, \lambda) = R_Z(\phi)R_Y(\pi/2)R_Z(\lambda)`.
Represented by matrix :math:`\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & e^{-i\lambda} \\
e^{i\phi} & e^{i(\phi+\lambda)} \end{pmatrix}`
"""


class U3Factory:
    name: Literal["U3"] = gate_names.U3

    def __call__(
        self, target_index: int, theta: float, phi: float, lmd: float
    ) -> QuantumGate:
        return QuantumGate(
            name=self.name, target_indices=(target_index,), params=(theta, phi, lmd)
        )


U3 = U3Factory()
r"""U3 gate is a generic single-qubit rotation gate with 3 Euler angles.
Represented by matrix
:math:`\begin{pmatrix}
\cos\frac{\theta}{2} & -e^{-i\lambda}\sin\frac{\theta}{2} \\
e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2}
\end{pmatrix}`
"""


class CNOTFactory:
    name: Literal["CNOT"] = gate_names.CNOT

    def __call__(self, control_index: int, target_index: int) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index,),
            control_indices=(control_index,),
        )


CNOT = CNOTFactory()
r"""CNOT gate, also called controlled-X gate, or CX gate CNOT(control, target) =
:math:`I \otimes |0\rangle\langle0| + X \otimes |1\rangle\langle1|`
"""


class CZFactory:
    name: Literal["CZ"] = gate_names.CZ

    def __call__(self, control_index: int, target_index: int) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index,),
            control_indices=(control_index,),
        )


CZ = CZFactory()
r"""CZ gate, also called controlled-Z gate CZ(control, target) =
:math:`I \otimes |0\rangle\langle0| + Z \otimes |1\rangle\langle1|`
"""


class SWAPFactory:
    name: Literal["SWAP"] = gate_names.SWAP

    def __call__(self, target_index1: int, target_index2: int) -> QuantumGate:
        return QuantumGate(
            name=self.name, target_indices=(target_index1, target_index2)
        )


SWAP = SWAPFactory()
"""SWAP gate."""


class TOFFOLIFactory:
    name: Literal["TOFFOLI"] = gate_names.TOFFOLI

    def __call__(
        self, control_index1: int, control_index2: int, target_index: int
    ) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=(target_index,),
            control_indices=(control_index1, control_index2),
        )


TOFFOLI = TOFFOLIFactory()
"""TOFFOLI gate."""


class UnitaryMatrixFactory:
    name: Literal["UnitaryMatrix"] = gate_names.UnitaryMatrix

    def __call__(
        self, target_indices: Sequence[int], unitary_matrix: Sequence[Sequence[complex]]
    ) -> QuantumGate:
        n = 2 ** len(target_indices)
        arr = np.array(unitary_matrix)
        if arr.shape != (n, n):
            raise ValueError(
                "The number of qubits does not match the size of the unitary matrix."
            )
        if not np.allclose(np.matmul(arr, np.conj(arr.T)), np.identity(n)):
            raise ValueError("The given matrix is not unitary.")
        return QuantumGate(
            name=self.name,
            target_indices=tuple(target_indices),
            unitary_matrix=tuple(map(tuple, unitary_matrix)),
        )


UnitaryMatrix = UnitaryMatrixFactory()
"""UnitaryMatrix gate represented by an arbitrary unitary matrix."""


class SingleQubitUnitaryMatrixFactory:
    name: Literal["UnitaryMatrix"] = gate_names.UnitaryMatrix

    def __call__(
        self, target_index: int, unitary_matrix: Sequence[Sequence[complex]]
    ) -> QuantumGate:
        return UnitaryMatrix((target_index,), unitary_matrix)


SingleQubitUnitaryMatrix = SingleQubitUnitaryMatrixFactory()
"""Single qubit UnitaryMatrix gate."""


class TwoQubitUnitaryMatrixFactory:
    name: Literal["UnitaryMatrix"] = gate_names.UnitaryMatrix

    def __call__(
        self,
        target_index1: int,
        target_index2: int,
        unitary_matrix: Sequence[Sequence[complex]],
    ) -> QuantumGate:
        return UnitaryMatrix((target_index1, target_index2), unitary_matrix)


TwoQubitUnitaryMatrix = TwoQubitUnitaryMatrixFactory()
"""Two qubit UnitaryMatrix gate."""


class PauliFactory:
    name: Literal["Pauli"] = gate_names.Pauli

    def __call__(
        self, target_indices: Sequence[int], pauli_ids: Sequence[int]
    ) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=tuple(target_indices),
            pauli_ids=tuple(pauli_ids),
        )


Pauli = PauliFactory()
"""Multi-qubit Pauli gate, consists of Pauli X, Y, or Z gates."""


class PauliRotationFactory:
    name: Literal["PauliRotation"] = gate_names.PauliRotation

    def __call__(
        self, target_indices: Sequence[int], pauli_ids: Sequence[int], angle: float
    ) -> QuantumGate:
        return QuantumGate(
            name=self.name,
            target_indices=tuple(target_indices),
            pauli_ids=tuple(pauli_ids),
            params=(angle,),
        )


PauliRotation = PauliRotationFactory()
"""Multi-qubit Pauli rotation gate such as :math:`e^{-iX_0Y_1 \\phi / 2}`."""


class ParametricRXFactory:
    name: Literal["ParametricRX"] = gate_names.ParametricRX

    def __call__(self, target_index: int) -> ParametricQuantumGate:
        return ParametricQuantumGate(name=self.name, target_indices=(target_index,))


ParametricRX = ParametricRXFactory()
"""Parametric RX gate.

Note that the instance of this class doesn't contain parameter values.
Every parametric gate is carried with it's parameter
(:class:`~Parameter`) such as (ParametricRX, Parameter).
"""


class ParametricRYFactory:
    name: Literal["ParametricRY"] = gate_names.ParametricRY

    def __call__(self, target_index: int) -> ParametricQuantumGate:
        return ParametricQuantumGate(name=self.name, target_indices=(target_index,))


ParametricRY = ParametricRYFactory()
"""Parametric RY gate.

Note that the instance of this class doesn't contain parameter values.
Every parametric gate is carried with it's parameter
(:class:`~Parameter`) such as (ParametricRY, Parameter).
"""


class ParametricRZFactory:
    name: Literal["ParametricRZ"] = gate_names.ParametricRZ

    def __call__(self, target_index: int) -> ParametricQuantumGate:
        return ParametricQuantumGate(name=self.name, target_indices=(target_index,))


ParametricRZ = ParametricRZFactory()
"""Parametric RZ gate.

Note that the instance of this class doesn't contain parameter values.
Every parametric gate is carried with it's parameter
(:class:`~Parameter`) such as (ParametricRZ, Parameter).
"""


class ParametricPauliRotationFactory:
    name: Literal["ParametricPauliRotation"] = gate_names.ParametricPauliRotation

    def __call__(
        self, target_indics: Sequence[int], pauli_ids: Sequence[int]
    ) -> ParametricQuantumGate:
        return ParametricQuantumGate(
            name=self.name,
            target_indices=tuple(target_indics),
            pauli_ids=tuple(pauli_ids),
        )


ParametricPauliRotation = ParametricPauliRotationFactory()
"""Parametric Pauli rotation gate.

Note that the instance of this class doesn't contain parameter values.
Every parametric gate is carried with it's parameter
(:class:`~Parameter`) such as (ParametricPauliRotation, Parameter).
"""
