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
from typing import cast

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


def _decompose_product_state(
    product_state: npt.NDArray[np.complex128], eps: float = 1e-10
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    argmax = np.argmax(np.abs(product_state))
    a, b = np.zeros(2, dtype=np.complex128), np.zeros(2, dtype=np.complex128)

    if abs(product_state[0]) < eps and abs(product_state[1]) < eps:
        a[:] = 0, 1
    else:
        if argmax == 0 or argmax == 2:
            c = product_state[2] / product_state[0]
        else:
            c = product_state[3] / product_state[1]
        a[0] = np.sqrt(1 / (1 + abs(c) ** 2))

    if abs(product_state[2]) < eps and abs(product_state[3]) < eps:
        a[:] = 1, 0
    else:
        if argmax == 0 or argmax == 2:
            c = product_state[0] / product_state[2]
        else:
            c = product_state[1] / product_state[3]
        ca = -cmath.phase(c)
        a[1] = np.sqrt(1 / (1 + abs(c) ** 2)) * (math.cos(ca) + 1j * math.sin(ca))

    if abs(product_state[0]) < eps and abs(product_state[2]) < eps:
        b[:] = 0, 1
    else:
        if argmax == 0 or argmax == 1:
            c = product_state[1] / product_state[0]
        else:
            c = product_state[3] / product_state[2]
        b[0] = np.sqrt(1 / (1 + abs(c) ** 2))

    if abs(product_state[1]) < eps and abs(product_state[3]) < eps:
        b[:] = 1, 0
    else:
        if argmax == 0 or argmax == 1:
            c = product_state[0] / product_state[1]
        else:
            c = product_state[2] / product_state[3]
        ca = -cmath.phase(c)
        b[1] = np.sqrt(1 / (1 + abs(c) ** 2)) * (math.cos(ca) + 1j * math.sin(ca))

    ab = np.kron(a, b)
    nonzero_idx = np.argmax(np.abs(ab))
    a *= product_state[nonzero_idx] / ab[nonzero_idx]
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    return a, b


def _psi_ext(
    psi: npt.NDArray[np.complex128], prime_flag: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    mu = (psi.T[0] + 1j * psi.T[1]) / np.sqrt(2)
    nu = (psi.T[0] - 1j * psi.T[1]) / np.sqrt(2)
    a, b = _decompose_product_state(mu)
    a_bar, b_bar = _decompose_product_state(nu)

    nonzero_idx = np.argmax(np.abs(np.kron(a, b_bar)))
    nonzero_idx2 = np.argmax(np.abs(np.kron(a_bar, b)))
    exp_delta = (psi.T[2][nonzero_idx] + 1j * psi.T[3][nonzero_idx]) / (
        np.sqrt(2) * np.kron(a, b_bar)[nonzero_idx]
    )
    exp_delta2 = (psi.T[2][nonzero_idx2] + 1j * psi.T[3][nonzero_idx2]) / (
        np.sqrt(2) * np.kron(a_bar, b)[nonzero_idx2]
    )
    if abs(1 - abs(exp_delta)) > abs(1 - abs(exp_delta2)):
        a, a_bar = a_bar, a
        b, b_bar = b_bar, b
        exp_delta = exp_delta2
    exp_delta /= abs(exp_delta)

    zero_ket, one_ket = np.array([[1], [0]]), np.array([[0], [1]])
    r_i = np.kron(zero_ket, np.conj(a)) + exp_delta * np.kron(one_ket, np.conj(a_bar))
    r_j = np.kron(zero_ket, np.conj(b)) + np.conj(exp_delta) * np.kron(
        one_ket, np.conj(b_bar)
    )

    xi = np.zeros(4)
    r_ij = np.kron(r_i, r_j)
    xi[0] = -cmath.phase(np.inner(r_ij[0], psi.T[0]))
    xi[1] = -cmath.phase(np.inner(r_ij[0], psi.T[1]) / -1j)
    xi[2] = -cmath.phase(np.inner(r_ij[1], psi.T[2]) / -1)
    xi[3] = -cmath.phase(np.inner(r_ij[1], psi.T[3]) / -1j)

    if prime_flag:
        r_i, r_j = np.conj(r_i.T), np.conj(r_j.T)

    theta_i, theta_j = su2_decompose(r_i), su2_decompose(r_j)

    return theta_i, theta_j, xi


def _all_different(xs: npt.NDArray[np.complex128], error: float = 1.0e-9) -> bool:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if abs(xs[i] - xs[j]) < error:
                return False
    return True


def su4_decompose(
    ut: Sequence[Sequence[complex]],
    eiglim: float = 1e-10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    m = np.array(
        [[1, -1j, 0, 0], [0, 0, -1, -1j], [0, 0, 1, -1j], [1, 1j, 0, 0]]
    ) / np.sqrt(2)
    m_inv = np.conj(m.T)
    u_prime = m_inv @ ut @ m
    w = u_prime.T @ u_prime
    if abs(np.sum(np.abs(np.diag(w))) - 4) < eiglim:
        eigvalue, eigvec = np.diag(w), np.identity(4, dtype=np.complex128)
    else:
        eigvalue, eigvec = np.linalg.eig(w)

        if not _all_different(eigvalue):
            raise ValueError("Dupliation of eigenvalues for SU4_decompose()")

        eigvec_t = np.zeros((4, 4), dtype=np.complex128)
        for i in range(4):
            u = eigvec.T[i]
            for j in range(i):
                u -= np.vdot(eigvec_t[j], u) * eigvec_t[j]
            nonzero_idx = cast(int, np.argmax(np.abs(u)))
            eigvec_t[i] = (
                u / (u[nonzero_idx] / np.abs(u[nonzero_idx])) / np.linalg.norm(u)
            )
        eigvec = eigvec_t.T

    eps = [cmath.phase(eigvalue[i]) / 2 for i in range(4)]
    psi = m @ eigvec
    theta_i, theta_j, xi = _psi_ext(psi, False)
    psi_prime = (
        ut @ psi @ np.diag([math.cos(eps[i]) - 1j * math.sin(eps[i]) for i in range(4)])
    )
    theta_prime_i, theta_prime_j, xi_prime = _psi_ext(psi_prime, True)

    b = xi_prime - xi - eps
    alpha = np.zeros(4)
    alpha[0] = -(b[0] + b[1] + b[2] + b[3]) / 4
    alpha[1] = -(b[0] - b[1] - b[2] + b[3]) / 4
    alpha[2] = -(-b[0] + b[1] - b[2] + b[3]) / 4
    alpha[3] = -(b[0] + b[1] - b[2] - b[3]) / 4

    return (
        alpha[1:],
        np.append(theta_i[1:], theta_j[1:]).reshape(2, 3),
        np.append(theta_prime_i[1:], theta_prime_j[1:]).reshape(2, 3),
    )


class SingleQubitUnitaryMatrix2RYRZTranspiler(GateDecomposer):
    """CircuitTranspiler, which decomposes single qubit UnitaryMatrix gates
    into gate sequences containing RY and RZ gates.

    Ref:
        [1]: Tomonori Shirakawa, Hiroshi Ueda, and Seiji Yunoki,
            Automatic quantum circuit encoding of a given arbitrary quantum state,
            arXiv:2112.14524v1, p.22, (2021).
        [2]: Main functions are ported from the source code wrtitten by Sota Morisaki
            in QunaSys intern project.
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


class TwoQubitUnitaryMatrixKAKTranspiler(GateDecomposer):
    """CircuitTranspiler, which decomposes two qubit UnitaryMatrix gates into
    gate sequences containing H, S, RX, RY, RZ, and CNOT gates.

    Raises:
        ValueError: Depending on the nature of the unitary matrix of the input
            UnitaryMatrix gate, the decomposition may fail and throw an error.

    Ref:
        [1]: Tomonori Shirakawa, Hiroshi Ueda, and Seiji Yunoki,
            Automatic quantum circuit encoding of a given arbitrary quantum state,
            arXiv:2112.14524v1, pp.4-5, (2021).
        [2]: Main functions are ported from the source code wrtitten by Sota Morisaki
            in QunaSys intern project.
    """

    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.name == gate_names.UnitaryMatrix and len(gate.target_indices) == 2

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        alpha, xi, xi_prime = su4_decompose(gate.unitary_matrix)
        q0, q1 = gate.target_indices

        return [
            gates.RZ(q1, xi[0][0]),
            gates.RY(q1, xi[0][1]),
            gates.RZ(q1, xi[0][2]),
            gates.RZ(q0, xi[1][0]),
            gates.RY(q0, xi[1][1]),
            gates.RZ(q0, xi[1][2]),
            gates.CNOT(q1, q0),
            gates.RX(q1, -2.0 * alpha[0]),
            gates.H(q1),
            gates.RZ(q0, -2.0 * alpha[2]),
            gates.CNOT(q1, q0),
            gates.S(q1),
            gates.H(q1),
            gates.RZ(q0, 2.0 * alpha[1]),
            gates.CNOT(q1, q0),
            gates.RX(q1, -np.pi / 2.0),
            gates.RZ(q1, xi_prime[0][0]),
            gates.RY(q1, xi_prime[0][1]),
            gates.RZ(q1, xi_prime[0][2]),
            gates.RX(q0, np.pi / 2.0),
            gates.RZ(q0, xi_prime[1][0]),
            gates.RY(q0, xi_prime[1][1]),
            gates.RZ(q0, xi_prime[1][2]),
        ]
