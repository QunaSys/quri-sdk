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
import numpy.typing as npt
import tensornetwork as tn
from numpy.testing import assert_almost_equal

from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.gates import CNOT, RY, U1, U2, U3, H, X, Y, Z
from quri_parts.tensornetwork.circuit import convert_circuit

circuit_tensor_pairs = [
    (
        QuantumCircuit(1, gates=[X(0)]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ),
    (
        QuantumCircuit(1, gates=[Y(0)]),
        np.array([[0.0, 1.0j], [-1.0j, 0.0]]),
    ),
    (
        QuantumCircuit(1, gates=[Z(0)]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
    ),
    (
        QuantumCircuit(2, gates=[H(0), CNOT(0, 1)]),
        np.array(
            [
                [
                    [
                        [1 / np.sqrt(2), 0.0],
                        [0.0, 1 / np.sqrt(2)],
                    ],
                    [
                        [0.0, 1 / np.sqrt(2)],
                        [1 / np.sqrt(2), 0.0],
                    ],
                ],
                [
                    [
                        [1 / np.sqrt(2), 0.0],
                        [0.0, -1 / np.sqrt(2)],
                    ],
                    [
                        [0.0, 1 / np.sqrt(2)],
                        [-1 / np.sqrt(2), 0.0],
                    ],
                ],
            ]
        ),
    ),
]


def test_convert_circuit() -> None:
    for c, t in circuit_tensor_pairs:
        tensornetwork_circuit = convert_circuit(c)
        all_edges = list(tensornetwork_circuit.input_edges) + list(
            tensornetwork_circuit.output_edges
        )
        contracted_state = tn.contractors.optimal(
            tensornetwork_circuit._container, output_edge_order=all_edges
        )
        assert_almost_equal(contracted_state.tensor, t)


def _ry_matrix(theta: float) -> npt.NDArray[np.complex128]:
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    )


def _u1_matrix(lam: float) -> npt.NDArray[np.complex128]:
    return np.diag([1.0, np.exp(1j * lam)]).astype(np.complex128)


def _u2_matrix(phi: float, lam: float) -> npt.NDArray[np.complex128]:
    return np.array(  # type: ignore
        [
            [1.0, -np.exp(1j * lam)],
            [np.exp(1j * phi), np.exp(1j * (phi + lam))],
        ],
        dtype=np.complex128,
    ) / np.sqrt(2)


def _u3_matrix(theta: float, phi: float, lam: float) -> npt.NDArray[np.complex128]:
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ],
        dtype=np.complex128,
    )


_H_MATRIX = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2)
# CNOT with control = qubit 0 (least significant bit), target = qubit 1
_CNOT01_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.complex128
)
_I2 = np.eye(2, dtype=np.complex128)

# Circuits using non-symmetric gate matrices, paired with the statevector
# obtained by plain matrix products (qubit 0 = least significant bit).
# Regression test: a gate tensor stored untransposed (matrix form instead of
# tensor form) applies the transposed gate, which is invisible for symmetric
# matrices like H/X/CNOT and, on product states, even for RY probabilities —
# it only shows up when a non-symmetric gate acts on an entangled state.
_theta, _phi, _lam = 0.9, 0.55, 1.3
circuit_state_pairs = [
    (
        QuantumCircuit(2, gates=[RY(0, 0.3), RY(1, 0.4), CNOT(0, 1), RY(0, _theta)]),
        np.kron(_I2, _ry_matrix(_theta))
        @ _CNOT01_MATRIX
        @ np.kron(_ry_matrix(0.4), _ry_matrix(0.3))
        @ np.array([1, 0, 0, 0], dtype=np.complex128),
    ),
    (
        QuantumCircuit(1, gates=[H(0), U1(0, _lam)]),
        _u1_matrix(_lam) @ _H_MATRIX @ np.array([1, 0], dtype=np.complex128),
    ),
    (
        QuantumCircuit(1, gates=[H(0), U2(0, _phi, _lam)]),
        _u2_matrix(_phi, _lam) @ _H_MATRIX @ np.array([1, 0], dtype=np.complex128),
    ),
    (
        QuantumCircuit(1, gates=[H(0), U3(0, _theta, _phi, _lam)]),
        _u3_matrix(_theta, _phi, _lam)
        @ _H_MATRIX
        @ np.array([1, 0], dtype=np.complex128),
    ),
]


def test_convert_circuit_nonsymmetric_gates() -> None:
    for c, expected in circuit_state_pairs:
        tensornetwork_circuit = convert_circuit(c)
        all_edges = list(tensornetwork_circuit.input_edges) + list(
            tensornetwork_circuit.output_edges
        )
        contracted = tn.contractors.optimal(
            tensornetwork_circuit._container, output_edge_order=all_edges
        )
        n = c.qubit_count
        # Fix all input axes to |0> and reorder output axes so that qubit 0 is
        # the least significant bit of the raveled statevector.
        state_tensor = contracted.tensor[(0,) * n]
        statevector = np.transpose(state_tensor, tuple(range(n))[::-1]).reshape(-1)
        assert_almost_equal(statevector, expected)
