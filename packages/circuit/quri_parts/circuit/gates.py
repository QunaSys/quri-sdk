# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Literal

from typing_extensions import deprecated

from quri_parts.circuit import gate_names
from quri_parts.rust.circuit.gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    H,
    Identity,
    Measurement,
    ParametricPauliRotation,
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    Pauli,
    PauliRotation,
    S,
    Sdag,
    SingleQubitUnitaryMatrix,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    TwoQubitUnitaryMatrix,
    UnitaryMatrix,
    X,
    Y,
    Z,
)

from .gate import ParametricQuantumGate, QuantumGate

#: Hadamard gate represented by matrix
#: :math:`\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}`
H = H

#: S Gate, or sqrt(Z) gate represented by matrix
#: :math:`\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}
S = S
#: Sdag Gate, conjugate of S gate represented by matrix
#: :math:`\begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}`
Sdag = Sdag
#: Square root of X gate represented by matrix
#: :math:`\frac{1}{2}\begin{pmatrix} 1+i & 1-i \\ 1-i & 1+i \end{pmatrix}`
SqrtX = SqrtX

#: Conjugate of the quare root of X gate represented by matrix
#: :math:`\frac{1}{2}\begin{pmatrix} 1-i & 1+i \\ 1+i & 1-i \end{pmatrix}`
SqrtXdag = SqrtXdag

#: Square root of Y gate represented by matrix
#: :math:`\frac{1+i}{2}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}`
SqrtY = SqrtY
#: Conjugate of the quare root of Y gate represented by matrix
#: :math:`\frac{1-i}{2}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}`
SqrtYdag = SqrtYdag

#: T gate, or sqrt(S) gate represented by matrix
#: :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}`
T = T

#: Conjugate of the T gate represented by matrix
#: :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}`
Tdag = Tdag
#: RX gate equivalant to :math:`\exp(-i\theta X/2)` represented by matrix
#: :math:`\begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
#: -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}`
RX = RX

#: RY gate equivalant to :math:`\exp(-i\theta Y/2)` represented by matrix
#: :math:`\begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
#: \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}`
RY = RY

#: RZ gate equivalant to :math:`\exp(-i\theta Z/2)` represented by matrix
#: :math:`\begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}`
RZ = RZ
#: U1 gate is a single-qubit rotation about the Z axis:
#:
#: :math:`U_1(\lambda) = e^{i\lambda/2} R_Z(\lambda)`.
#: Represented by matrix :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{pmatrix}`
U1 = U1
#: U2 gate is a single-qubit rotation about X + Z axis:
#:
#: :math:`U_2(\phi, \lambda) = R_Z(\phi)R_Y(\pi/2)R_Z(\lambda)`.
#: Represented by matrix :math:`\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -e^{i\lambda} \\
#: e^{i\phi} & e^{i(\phi+\lambda)} \end{pmatrix}`
U2 = U2
#: U3 gate is a generic single-qubit rotation gate with 3 Euler angles.
#: Represented by matrix
#: :math:`\begin{pmatrix}
#: \cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\
#: e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2}
#: \end{pmatrix}`
U3 = U3
#: CNOT gate, also called controlled-X gate, or CX gate CNOT(control, target) =
#: :math:`I \otimes |0\rangle\langle0| + X \otimes |1\rangle\langle1|`
CNOT = CNOT
#: CZ gate, also called controlled-Z gate CZ(control, target) =
#: :math:`I \otimes |0\rangle\langle0| + Z \otimes |1\rangle\langle1|`
CZ = CZ
#: SWAP gate.
SWAP = SWAP
#: TOFFOLI gate.
TOFFOLI = TOFFOLI
#: UnitaryMatrix gate represented by an arbitrary unitary matrix.
UnitaryMatrix = UnitaryMatrix
#: Single qubit UnitaryMatrix gate.
SingleQubitUnitaryMatrix = SingleQubitUnitaryMatrix
#: Two qubit UnitaryMatrix gate.
TwoQubitUnitaryMatrix = TwoQubitUnitaryMatrix
#: Multi-qubit Pauli gate, consists of Pauli X, Y, or Z gates.
Pauli = Pauli
#: Multi-qubit Pauli rotation gate such as :math:`e^{-iX_0Y_1 \\phi / 2}`.
PauliRotation = PauliRotation
#: Parametric RX gate.
#:
#: Note that the instance of this class doesn't contain parameter values.
#: Every parametric gate is carried with it's parameter
#: (:class:`~Parameter`) such as (ParametricRX, Parameter).
ParametricRX = ParametricRX
#: Parametric RY gate.
#:
#: Note that the instance of this class doesn't contain parameter values.
#: Every parametric gate is carried with it's parameter
#: (:class:`~Parameter`) such as (ParametricRY, Parameter).
ParametricRY = ParametricRY
#: Parametric RZ gate.
#:
#: Note that the instance of this class doesn't contain parameter values.
#: Every parametric gate is carried with it's parameter
#: (:class:`~Parameter`) such as (ParametricRZ, Parameter).
ParametricRZ = ParametricRZ
#: Parametric Pauli rotation gate.
#:
#: Note that the instance of this class doesn't contain parameter values.
#: Every parametric gate is carried with it's parameter
#: (:class:`~Parameter`) such as (ParametricPauliRotation, Parameter).
ParametricPauliRotation = ParametricPauliRotation
#: Measurement gate that transfers the measurement result to a classical
#: bit.
Measurement = Measurement
#: Identity gate represented by matrix
#: :math:`\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}`
Identity = Identity
# Pauli Z gate represented by matrix
# :math:`\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}`
Z = Z
#: Pauli Y gate represented by matrix
#: :math:`\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}`
Y = Y
#: Pauli X gate represented by matrix
#: :math:`\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`
X = X


@deprecated("Factory class is deprecated")
class IdentityFactory:
    name: Literal["Identity"] = gate_names.Identity

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return Identity(*args, **kwargs)


@deprecated("Factory class is deprecated")
class XFactory:
    name: Literal["X"] = gate_names.X

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return X(*args, **kwargs)


@deprecated("Factory class is deprecated")
class YFactory:
    name: Literal["Y"] = gate_names.Y

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return Y(*args, **kwargs)


@deprecated("Factory class is deprecated")
class ZFactory:
    name: Literal["Z"] = gate_names.Z

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return Z(*args, **kwargs)


@deprecated("Factory class is deprecated")
class HFactory:
    name: Literal["H"] = gate_names.H

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return H(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SFactory:
    name: Literal["S"] = gate_names.S

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return S(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SdagFactory:
    name: Literal["Sdag"] = gate_names.Sdag

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return Sdag(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SqrtXFactory:
    name: Literal["SqrtX"] = gate_names.SqrtX

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return SqrtX(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SqrtXdagFactory:
    name: Literal["SqrtXdag"] = gate_names.SqrtXdag

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return SqrtXdag(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SqrtYFactory:
    name: Literal["SqrtY"] = gate_names.SqrtY

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return SqrtY(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SqrtYdagFactory:
    name: Literal["SqrtYdag"] = gate_names.SqrtYdag

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return SqrtYdag(*args, **kwargs)


@deprecated("Factory class is deprecated")
class TFactory:
    name: Literal["T"] = gate_names.T

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return T(*args, **kwargs)


@deprecated("Factory class is deprecated")
class TdagFactory:
    name: Literal["Tdag"] = gate_names.Tdag

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return Tdag(*args, **kwargs)


@deprecated("Factory class is deprecated")
class RXFactory:
    name: Literal["RX"] = gate_names.RX

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return RX(*args, **kwargs)


@deprecated("Factory class is deprecated")
class RYFactory:
    name: Literal["RY"] = gate_names.RY

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return RY(*args, **kwargs)


@deprecated("Factory class is deprecated")
class RZFactory:
    name: Literal["RZ"] = gate_names.RZ

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return RZ(*args, **kwargs)


@deprecated("Factory class is deprecated")
class U1Factory:
    name: Literal["U1"] = gate_names.U1

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return U1(*args, **kwargs)


@deprecated("Factory class is deprecated")
class U2Factory:
    name: Literal["U2"] = gate_names.U2

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return U2(*args, **kwargs)


@deprecated("Factory class is deprecated")
class U3Factory:
    name: Literal["U3"] = gate_names.U3

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return U3(*args, **kwargs)


@deprecated("Factory class is deprecated")
class CNOTFactory:
    name: Literal["CNOT"] = gate_names.CNOT

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return CNOT(*args, **kwargs)


@deprecated("Factory class is deprecated")
class CZFactory:
    name: Literal["CZ"] = gate_names.CZ

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return CZ(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SWAPFactory:
    name: Literal["SWAP"] = gate_names.SWAP

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return SWAP(*args, **kwargs)


@deprecated("Factory class is deprecated")
class TOFFOLIFactory:
    name: Literal["TOFFOLI"] = gate_names.TOFFOLI

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return TOFFOLI(*args, **kwargs)


@deprecated("Factory class is deprecated")
class UnitaryMatrixFactory:
    name: Literal["UnitaryMatrix"] = gate_names.UnitaryMatrix

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return UnitaryMatrix(*args, **kwargs)


@deprecated("Factory class is deprecated")
class SingleQubitUnitaryMatrixFactory:
    name: Literal["UnitaryMatrix"] = gate_names.UnitaryMatrix

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return UnitaryMatrix(*args, **kwargs)


@deprecated("Factory class is deprecated")
class TwoQubitUnitaryMatrixFactory:
    name: Literal["UnitaryMatrix"] = gate_names.UnitaryMatrix

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return UnitaryMatrix(*args, **kwargs)


@deprecated("Factory class is deprecated")
class PauliFactory:
    name: Literal["Pauli"] = gate_names.Pauli

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return Pauli(*args, **kwargs)


@deprecated("Factory class is deprecated")
class PauliRotationFactory:
    name: Literal["PauliRotation"] = gate_names.PauliRotation

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return PauliRotation(*args, **kwargs)


@deprecated("Factory class is deprecated")
class ParametricRXFactory:
    name: Literal["ParametricRX"] = gate_names.ParametricRX

    def __call__(self, *args: Any, **kwargs: Any) -> ParametricQuantumGate:
        return ParametricRX(*args, **kwargs)


@deprecated("Factory class is deprecated")
class ParametricRYFactory:
    name: Literal["ParametricRY"] = gate_names.ParametricRY

    def __call__(self, *args: Any, **kwargs: Any) -> ParametricQuantumGate:
        return ParametricRY(*args, **kwargs)


@deprecated("Factory class is deprecated")
class ParametricRZFactory:
    name: Literal["ParametricRZ"] = gate_names.ParametricRZ

    def __call__(self, *args: Any, **kwargs: Any) -> ParametricQuantumGate:
        return ParametricRZ(*args, **kwargs)


@deprecated("Factory class is deprecated")
class ParametricPauliRotationFactory:
    name: Literal["ParametricPauliRotation"] = gate_names.ParametricPauliRotation

    def __call__(self, *args: Any, **kwargs: Any) -> ParametricQuantumGate:
        return ParametricPauliRotation(*args, **kwargs)


@deprecated("Factory class is deprecated")
class MeasurementFactory:
    name: Literal["Measurement"] = gate_names.Measurement

    def __call__(self, *args: Any, **kwargs: Any) -> QuantumGate:
        return Measurement(*args, **kwargs)


__all__ = [
    "XFactory",
    "YFactory",
    "ZFactory",
    "IdentityFactory",
    "MeasurementFactory",
    "ParametricPauliRotationFactory",
    "ParametricRZFactory",
    "ParametricRYFactory",
    "ParametricRXFactory",
    "PauliRotationFactory",
    "PauliFactory",
    "TwoQubitUnitaryMatrixFactory",
    "SingleQubitUnitaryMatrixFactory",
    "UnitaryMatrixFactory",
    "TOFFOLIFactory",
    "SWAPFactory",
    "CZFactory",
    "CNOTFactory",
    "U3Factory",
    "U2Factory",
    "U1Factory",
    "RZFactory",
    "RYFactory",
    "RXFactory",
    "TdagFactory",
    "TFactory",
    "SqrtYdagFactory",
    "SqrtYFactory",
    "SqrtXdagFactory",
    "SqrtXFactory",
    "SdagFactory",
    "SFactory",
    "HFactory",
    "X",
    "Y",
    "Z",
    "Identity",
    "Measurement",
    "ParametricPauliRotation",
    "ParametricRZ",
    "ParametricRY",
    "ParametricRX",
    "PauliRotation",
    "Pauli",
    "TwoQubitUnitaryMatrix",
    "SingleQubitUnitaryMatrix",
    "UnitaryMatrix",
    "TOFFOLI",
    "SWAP",
    "CZ",
    "CNOT",
    "U3",
    "U2",
    "U1",
    "RZ",
    "RY",
    "RX",
    "Tdag",
    "T",
    "SqrtYdag",
    "SqrtY",
    "SqrtXdag",
    "SqrtX",
    "Sdag",
    "S",
    "H",
]
