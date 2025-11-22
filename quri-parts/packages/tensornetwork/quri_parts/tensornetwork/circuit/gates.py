# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from copy import copy
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from tensornetwork import Edge, Node

"""The following gates are in tensor form, which should not be confused with their
matrix representation. Expanded in terms of its tensor representation can be written as

.. math::
    \\hat{O} = |i_{N},\\ldots,i_{2N-1}><i_0,\\ldots,i_{N-1}| T[i_0,\\ldots,i_{2N-1}]

Some of the below gates therefore appear to be transposed versions of their matrix
representations, but this is just a consequence of the tensor-arithmetic.
"""


_I_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, 1.0]]
_X_LIST: Sequence[Sequence[complex]] = [[0.0, 1.0], [1.0, 0.0]]
_Y_LIST: Sequence[Sequence[complex]] = [
    [0.0, 1.0j],
    [-1.0j, 0.0],
]
_Z_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, -1.0]]
_H_LIST: Sequence[Sequence[complex]] = [
    [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
    [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
]
_S_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, 1.0j]]
_SDAG_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, -1.0j]]
_SQRTX_LIST: Sequence[Sequence[complex]] = [
    [0.5 + 0.5j, 0.5 - 0.5j],
    [0.5 - 0.5j, 0.5 + 0.5j],
]
_SQRTXDAG_LIST: Sequence[Sequence[complex]] = [
    [0.5 - 0.5j, 0.5 + 0.5j],
    [0.5 + 0.5j, 0.5 - 0.5j],
]
_SQRTY_LIST: Sequence[Sequence[complex]] = [
    [0.5 + 0.5j, 0.5 + 0.5j],
    [-0.5 - 0.5j, 0.5 + 0.5j],
]
_SQRTYDAG_LIST: Sequence[Sequence[complex]] = [
    [0.5 - 0.5j, -0.5 + 0.5j],
    [0.5 - 0.5j, 0.5 - 0.5j],
]
_SQRTZ_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, 1.0j]]
_SQRTZDAG_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, -1.0j]]
_T_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, (1.0 + 1.0j) / np.sqrt(2)]]
_TDAG_LIST: Sequence[Sequence[complex]] = [[1.0, 0.0], [0.0, (1.0 - 1.0j) / np.sqrt(2)]]
_CNOT_LIST: Sequence[Sequence[Sequence[Sequence[complex]]]] = [
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ],
    [
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ],
]
_CZ_LIST: Sequence[Sequence[Sequence[Sequence[complex]]]] = [
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ],
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, -1.0]],
    ],
]
_SWAP_LIST: Sequence[Sequence[Sequence[Sequence[complex]]]] = [
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ],
    [
        [[0.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ],
]
_TOFFOLI_LIST: Sequence[Sequence[Sequence[Sequence[Sequence[Sequence[complex]]]]]] = [
    [
        [
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ],
        [
            [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ],
    ],
    [
        [
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
        ],
        [
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
        ],
    ],
]


class TensorNetworkQuantumGate(Node, ABC):  # type: ignore
    """:class:`~TensorNetworkQuantumGate` class is a base class that wraps
    :class:`~Node`."""

    def __init__(
        self,
        unitary_matrix: npt.NDArray[np.complex128],
        qubit_indices: Sequence[int],
        name: str,
        backend: str,
        conjugate: bool = False,
    ) -> None:
        self.conjugate = conjugate
        if conjugate:
            tensor = np.conj(unitary_matrix, dtype=np.complex128)
        else:
            tensor = unitary_matrix
        self.qubit_indices = qubit_indices
        super().__init__(tensor, name=name, backend=backend)

    @abstractmethod
    def copy(self, conjugate: bool = False) -> "TensorNetworkQuantumGate":
        pass

    @property
    @abstractmethod
    def input_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        pass

    @property
    @abstractmethod
    def output_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        pass


class SingleQubitGate(TensorNetworkQuantumGate, ABC):
    """:class:`~SingleQubitGate` class is a base class that facilitates one qubit
    gates."""

    def __init__(
        self,
        data: Sequence[Sequence[complex]],
        qubit_indices: Sequence[int],
        name: str,
        backend: str,
        conjugate: bool = False,
    ) -> None:
        assert len(qubit_indices) == 1
        tensor = np.array(data, dtype=np.complex128)
        super().__init__(
            tensor, qubit_indices, name=name, backend=backend, conjugate=conjugate
        )

    @property
    def input_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {self.qubit_indices[0]: self[0]}

    @property
    def output_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {self.qubit_indices[0]: self[1]}


class SingleQubitRotationGate(TensorNetworkQuantumGate, ABC):
    """:class:`~SingleQubitRotationGate` class is a base class that facilitates
    single qubit rotation gates."""

    @abstractmethod
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        pass

    def __init__(
        self,
        angles: Sequence[float],
        qubit_indices: Sequence[int],
        name: str,
        backend: str,
    ) -> None:
        assert len(qubit_indices) == 1
        self.angles = angles
        unitary_matrix = self.rotation(angles)
        super().__init__(unitary_matrix, qubit_indices, name=name, backend=backend)

    @property
    def input_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {self.qubit_indices[0]: self[0]}

    @property
    def output_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {self.qubit_indices[0]: self[1]}


class SingleQubitPauliRotationGate(SingleQubitRotationGate, ABC):
    """:class:`~SingleQubitPauliRotationGate` class is a base class that
    facilitates single qubit Pauli rotation gates."""

    pauli: Sequence[Sequence[complex]]

    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 1
        return np.transpose(
            np.cos(angles[0] / 2.0) * np.eye(2, dtype=np.complex128)
            - 1j * np.sin(angles[0] / 2.0) * np.array(self.pauli, dtype=np.complex128)
        )  # transpose is taken here to conform to tensor notation

    def __init__(
        self, angle: float, qubit_indices: Sequence[int], name: str, backend: str
    ) -> None:
        super().__init__([angle], qubit_indices, name=name, backend=backend)


class TwoQubitGate(TensorNetworkQuantumGate, ABC):
    """:class:`~TwoQubitGate` class is a base class that facilitates two qubit
    gates."""

    def __init__(
        self,
        data: Sequence[Sequence[Sequence[Sequence[complex]]]],
        qubit_indices: Sequence[int],
        name: str,
        backend: str,
        conjugate: bool = False,
    ) -> None:
        assert len(qubit_indices) == 2
        tensor = np.array(data, dtype=np.complex128)
        super().__init__(
            tensor, qubit_indices, name=name, backend=backend, conjugate=conjugate
        )

    @property
    def input_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {qi: self[i] for i, qi in enumerate(self.qubit_indices)}

    @property
    def output_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {qi: self[i + 2] for i, qi in enumerate(self.qubit_indices)}


class ThreeQubitGate(TensorNetworkQuantumGate, ABC):
    """:class:`~ThreeQubitGate` class is a base class that facilitates three
    qubit gates."""

    def __init__(
        self,
        data: Sequence[Sequence[Sequence[Sequence[Sequence[Sequence[complex]]]]]],
        qubit_indices: Sequence[int],
        name: str,
        backend: str,
        conjugate: bool = False,
    ) -> None:
        assert len(qubit_indices) == 3
        tensor = np.array(data, dtype=np.complex128)
        super().__init__(
            tensor, qubit_indices, name=name, backend=backend, conjugate=conjugate
        )

    @property
    def input_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {qi: self[i] for i, qi in enumerate(self.qubit_indices)}

    @property
    def output_qubit_edge_mapping(self) -> Mapping[int, Edge]:
        return {qi: self[i + 3] for i, qi in enumerate(self.qubit_indices)}


class I(SingleQubitGate):  # noqa: E742
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        unitary_matrix = _I_LIST
        name = "I"

        super().__init__(unitary_matrix, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "I":
        quantum_gate_copy = I(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class X(SingleQubitGate):
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        unitary_matrix = _X_LIST
        name = "X"

        super().__init__(unitary_matrix, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "X":
        quantum_gate_copy = X(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class Y(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _Y_LIST
        name = "Y"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "Y":
        quantum_gate_copy = Y(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class Z(SingleQubitGate):
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        unitary_matrix = _Z_LIST
        name = "Z"

        super().__init__(unitary_matrix, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "Z":
        quantum_gate_copy = Z(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class H(SingleQubitGate):
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        unitary_matrix = _H_LIST
        name = "H"

        super().__init__(unitary_matrix, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "H":
        quantum_gate_copy = H(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class S(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _S_LIST
        name = "S"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "S":
        quantum_gate_copy = S(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class Sdag(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _SDAG_LIST
        name = "Sdag"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "Sdag":
        quantum_gate_copy = Sdag(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class SqrtX(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _SQRTX_LIST
        name = "SqrtX"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "SqrtX":
        quantum_gate_copy = SqrtX(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class SqrtXdag(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _SQRTXDAG_LIST
        name = "SqrtXdag"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "SqrtXdag":
        quantum_gate_copy = SqrtXdag(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class SqrtY(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _SQRTY_LIST
        name = "SqrtY"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "SqrtY":
        quantum_gate_copy = SqrtY(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class SqrtYdag(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _SQRTYDAG_LIST
        name = "SqrtYdag"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "SqrtYdag":
        quantum_gate_copy = SqrtYdag(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class SqrtZ(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _SQRTZ_LIST
        name = "SqrtZ"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "SqrtZ":
        quantum_gate_copy = SqrtZ(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class SqrtZdag(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _SQRTZDAG_LIST
        name = "SqrtZdag"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "SqrtZdag":
        quantum_gate_copy = SqrtZdag(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class T(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _T_LIST
        name = "T"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "T":
        quantum_gate_copy = T(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class Tdag(SingleQubitGate):
    def __init__(
        self,
        qubit_indices: Sequence[int],
        backend: str = "numpy",
        conjugate: bool = False,
    ) -> None:
        unitary_matrix = _TDAG_LIST
        name = "Tdag"

        super().__init__(
            unitary_matrix, qubit_indices, name, backend, conjugate=conjugate
        )

    def copy(self, conjugate: bool = False) -> "Tdag":
        quantum_gate_copy = Tdag(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
            conjugate=conjugate,
        )
        return quantum_gate_copy


class Rx(SingleQubitPauliRotationGate):
    pauli = _X_LIST

    def __init__(
        self, angle: float, qubit_indices: Sequence[int], backend: str = "numpy"
    ) -> None:
        name = "RX"

        super().__init__(angle, qubit_indices, name, backend)

    def copy(self, conjugate: bool = False) -> "Rx":
        angle = -self.angles[0] if conjugate else self.angles[0]
        quantum_gate_copy = Rx(
            angle,
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class Ry(SingleQubitPauliRotationGate):
    pauli = _Y_LIST

    def __init__(
        self, angle: float, qubit_indices: Sequence[int], backend: str = "numpy"
    ) -> None:
        name = "RY"

        super().__init__(angle, qubit_indices, name, backend)

    def copy(self, conjugate: bool = False) -> "Ry":
        angle = -self.angles[0] if conjugate else self.angles[0]
        quantum_gate_copy = Ry(
            angle,
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class Rz(SingleQubitPauliRotationGate):
    pauli = _Z_LIST

    def __init__(
        self, angle: float, qubit_indices: Sequence[int], backend: str = "numpy"
    ) -> None:
        name = "RZ"

        super().__init__(angle, qubit_indices, name, backend)

    def copy(self, conjugate: bool = False) -> "Rz":
        angle = -self.angles[0] if conjugate else self.angles[0]
        quantum_gate_copy = Rz(
            angle,
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class U1(SingleQubitRotationGate):
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 1
        return np.array(
            [[1.0, 0.0], [0.0, np.exp(1j * angles[0])]], dtype=np.complex128
        )

    def __init__(
        self,
        angles: Sequence[float],
        qubit_indices: Sequence[int],
        backend: str = "numpy",
    ) -> None:
        name = "U1"

        super().__init__(angles, qubit_indices, name, backend)

    def copy(self, conjugate: bool = False) -> "U1":
        angles = [-a for a in self.angles] if conjugate else [a for a in self.angles]
        quantum_gate_copy = U1(
            angles,
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class U2(SingleQubitRotationGate):
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 2
        return np.array(
            [
                [1.0 / np.sqrt(2), np.exp(1j * angles[0]) / np.sqrt(2)],
                [
                    -np.exp(-1j * angles[1]) / np.sqrt(2),
                    np.exp(1j * (angles[0] + angles[1])) / np.sqrt(2),
                ],
            ],
            dtype=np.complex128,
        )

    def __init__(
        self,
        angles: Sequence[float],
        qubit_indices: Sequence[int],
        backend: str = "numpy",
    ) -> None:
        name = "U2"

        super().__init__(angles, qubit_indices, name, backend)

    def copy(self, conjugate: bool = False) -> "U2":
        angles = [-a for a in self.angles] if conjugate else [a for a in self.angles]
        quantum_gate_copy = U2(
            angles,
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class U3(SingleQubitRotationGate):
    def rotation(self, angles: Sequence[float]) -> npt.NDArray[np.complex128]:
        assert len(angles) == 3
        return np.array(
            [
                [
                    np.cos(angles[0] / 2.0),
                    np.exp(1j * angles[1]) * np.sin(angles[0] / 2.0),
                ],
                [
                    -np.exp(1j * angles[2]) * np.sin(angles[0] / 2.0),
                    np.exp(1j * (angles[1] + angles[2])) * np.cos(angles[0] / 2.0),
                ],
            ],
            dtype=np.complex128,
        )

    def __init__(
        self,
        angles: Sequence[float],
        qubit_indices: Sequence[int],
        backend: str = "numpy",
    ) -> None:
        name = "U3"

        super().__init__(angles, qubit_indices, name, backend)

    def copy(self, conjugate: bool = False) -> "U3":
        angles = [-a for a in self.angles] if conjugate else [a for a in self.angles]
        quantum_gate_copy = U3(
            angles,
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class CNOT(TwoQubitGate):
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        data = _CNOT_LIST
        name = "CNOT"

        super().__init__(data, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "CNOT":
        quantum_gate_copy = CNOT(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class CZ(TwoQubitGate):
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        data = _CZ_LIST
        name = "CZ"

        super().__init__(data, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "CZ":
        quantum_gate_copy = CZ(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class SWAP(TwoQubitGate):
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        data = _SWAP_LIST
        name = "SWAP"

        super().__init__(data, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "SWAP":
        quantum_gate_copy = SWAP(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy


class Toffoli(ThreeQubitGate):
    def __init__(self, qubit_indices: Sequence[int], backend: str = "numpy") -> None:
        data = _TOFFOLI_LIST
        name = "Toffoli"

        super().__init__(data, qubit_indices, name, backend)

    def copy(self, _: bool = False) -> "Toffoli":
        quantum_gate_copy = Toffoli(
            copy(list(self.input_qubit_edge_mapping.keys())),
            self.backend,
        )
        return quantum_gate_copy
