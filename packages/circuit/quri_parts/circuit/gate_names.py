# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, Union

from typing_extensions import TypeAlias, TypeGuard

SingleQubitGateNameType: TypeAlias = Literal[
    "Identity",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "Sdag",
    "SqrtX",
    "SqrtXdag",
    "SqrtY",
    "SqrtYdag",
    "T",
    "Tdag",
    "RX",
    "RY",
    "RZ",
    "U1",
    "U2",
    "U3",
]

Identity: Literal["Identity"] = "Identity"
X: Literal["X"] = "X"
Y: Literal["Y"] = "Y"
Z: Literal["Z"] = "Z"
H: Literal["H"] = "H"
S: Literal["S"] = "S"
Sdag: Literal["Sdag"] = "Sdag"
SqrtX: Literal["SqrtX"] = "SqrtX"
SqrtXdag: Literal["SqrtXdag"] = "SqrtXdag"
SqrtY: Literal["SqrtY"] = "SqrtY"
SqrtYdag: Literal["SqrtYdag"] = "SqrtYdag"
T: Literal["T"] = "T"
Tdag: Literal["Tdag"] = "Tdag"
RX: Literal["RX"] = "RX"
RY: Literal["RY"] = "RY"
RZ: Literal["RZ"] = "RZ"
U1: Literal["U1"] = "U1"
U2: Literal["U2"] = "U2"
U3: Literal["U3"] = "U3"

SINGLE_QUBIT_GATE_NAMES: set[SingleQubitGateNameType] = {
    Identity,
    X,
    Y,
    Z,
    H,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
}


def is_single_qubit_gate_name(gate_name: str) -> TypeGuard[SingleQubitGateNameType]:
    return gate_name in SINGLE_QUBIT_GATE_NAMES


TwoQubitGateNameType: TypeAlias = Literal[
    "CNOT",
    "CZ",
    "SWAP",
]

CNOT: Literal["CNOT"] = "CNOT"
CZ: Literal["CZ"] = "CZ"
SWAP: Literal["SWAP"] = "SWAP"

TWO_QUBIT_GATE_NAMES: set[TwoQubitGateNameType] = {CNOT, CZ, SWAP}


def is_two_qubit_gate_name(gate_name: str) -> TypeGuard[TwoQubitGateNameType]:
    return gate_name in TWO_QUBIT_GATE_NAMES


ThreeQubitGateNameType: TypeAlias = Literal[
    "TOFFOLI",
]

TOFFOLI: Literal["TOFFOLI"] = "TOFFOLI"

THREE_QUBIT_GATE_NAMES: set[ThreeQubitGateNameType] = {TOFFOLI}


def is_three_qubit_gate_name(gate_name: str) -> TypeGuard[ThreeQubitGateNameType]:
    return gate_name in THREE_QUBIT_GATE_NAMES


MultiQubitGateNameType: TypeAlias = Literal[
    "Pauli",
    "PauliRotation",
]

Pauli: Literal["Pauli"] = "Pauli"
PauliRotation: Literal["PauliRotation"] = "PauliRotation"

MULTI_QUBIT_GATE_NAMES: set[MultiQubitGateNameType] = {
    Pauli,
    PauliRotation,
}


def is_multi_qubit_gate_name(gate_name: str) -> TypeGuard[MultiQubitGateNameType]:
    return gate_name in MULTI_QUBIT_GATE_NAMES


#: A set of strings representing gate names.
NonParametricGateNameType: TypeAlias = Union[
    SingleQubitGateNameType, TwoQubitGateNameType, MultiQubitGateNameType
]


UnitaryMatrixGateNameType: TypeAlias = Literal[
    "UnitaryMatrix",
]

UnitaryMatrix: Literal["UnitaryMatrix"] = "UnitaryMatrix"

UNITARY_MATRIX_GATE_NAMES: set[UnitaryMatrixGateNameType] = {
    UnitaryMatrix,
}


def is_unitary_matrix_gate_name(gate_name: str) -> TypeGuard[UnitaryMatrixGateNameType]:
    return gate_name in UNITARY_MATRIX_GATE_NAMES


ParametricGateNameType: TypeAlias = Literal[
    "ParametricRX",
    "ParametricRY",
    "ParametricRZ",
    "ParametricPauliRotation",
]

ParametricRX: Literal["ParametricRX"] = "ParametricRX"
ParametricRY: Literal["ParametricRY"] = "ParametricRY"
ParametricRZ: Literal["ParametricRZ"] = "ParametricRZ"
ParametricPauliRotation: Literal["ParametricPauliRotation"] = "ParametricPauliRotation"

PARAMETRIC_GATE_NAMES: set[ParametricGateNameType] = {
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    ParametricPauliRotation,
}


def is_parametric_gate_name(gate_name: str) -> TypeGuard[ParametricGateNameType]:
    return gate_name in PARAMETRIC_GATE_NAMES


#: Valid Pauli gate names.
PauliNameType: TypeAlias = Literal["X", "Y", "Z", "Pauli"]

PAULI_NAMES: set[PauliNameType] = {
    X,
    Y,
    Z,
    Pauli,
}


def is_pauli_name(gate_name: str) -> TypeGuard[PauliNameType]:
    return gate_name in PAULI_NAMES


CliffordGateNameType: TypeAlias = Literal[
    "Identity",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "Sdag",
    "SqrtX",
    "SqrtXdag",
    "SqrtY",
    "SqrtYdag",
    "CNOT",
    "CZ",
    "SWAP",
    "Pauli",
]

CLIFFORD_GATE_NAMES: set[CliffordGateNameType] = {
    Identity,
    X,
    Y,
    Z,
    H,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    CNOT,
    CZ,
    SWAP,
    Pauli,
}

#: Valid gate names
GateNameType: TypeAlias = Literal[
    SingleQubitGateNameType,
    TwoQubitGateNameType,
    ThreeQubitGateNameType,
    MultiQubitGateNameType,
    UnitaryMatrixGateNameType,
    ParametricGateNameType,
]

GATE_NAMES: set[GateNameType] = (
    SINGLE_QUBIT_GATE_NAMES
    | TWO_QUBIT_GATE_NAMES
    | THREE_QUBIT_GATE_NAMES
    | MULTI_QUBIT_GATE_NAMES
    | UNITARY_MATRIX_GATE_NAMES
    | PARAMETRIC_GATE_NAMES
)


def is_gate_name(gate_name: str) -> TypeGuard[GateNameType]:
    return gate_name in GATE_NAMES
