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
from collections.abc import Sequence

import numpy as np

from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    gate_names,
    gates,
)

from .transpiler import (
    CircuitTranspilerProtocol,
    GateKindDecomposer,
    SequentialTranspiler,
)


class AdjacentGateFuser(CircuitTranspilerProtocol, ABC):
    """Abstract base class of adjacent gates fusing transpilers."""

    @property
    @abstractmethod
    def target_gate_count(self) -> int:
        ...

    @abstractmethod
    def is_target_sequence(self, seq: Sequence[QuantumGate]) -> bool:
        ...

    @abstractmethod
    def fuse(self, seq: Sequence[QuantumGate]) -> Sequence[QuantumGate]:
        ...

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        xs = list(circuit.gates)
        ys: list[QuantumGate] = []

        while len(xs) >= self.target_gate_count:
            ts = xs[: self.target_gate_count]

            if self.is_target_sequence(ts):
                xs[: self.target_gate_count] = self.fuse(ts)
            else:
                ys.append(xs[0])
                xs = xs[1:]

        ys.extend(xs)

        ret = QuantumCircuit(circuit.qubit_count)
        ret.extend(ys)
        return ret


class CNOTHCNOTFusingTranspiler(AdjacentGateFuser):
    """A CircuitTranspiler that reduces the number of CNOT gates by replacing
    partial circuits that satisfy certain conditions with equivalent circuits.

    The number of Clifford gates (here H, S, Sdag) and the circuit depth will increase,
    but the number of CNOT gates will decrease, which is expected to help reduce the
    error rate of the circuit in typical NISQ devices.

    Ref:
        Joshua Goings, et al., Molecular Symmetry in VQE: A Dual Approach for
        Trapped-Ion Simulations of Benzene, arXiv:2308.00667v1, pp.3-4, (2023).
    """

    @property
    def target_gate_count(self) -> int:
        return 3

    def is_target_sequence(self, seq: Sequence[QuantumGate]) -> bool:
        return (
            seq[0].name == gate_names.CNOT
            and seq[1].name == gate_names.H
            and seq[2].name == gate_names.CNOT
            and seq[0].control_indices == seq[2].control_indices
            and seq[0].target_indices == seq[2].target_indices
            and seq[0].control_indices == seq[1].target_indices
        )

    def fuse(self, seq: Sequence[QuantumGate]) -> Sequence[QuantumGate]:
        q0, q1 = seq[0].control_indices[0], seq[0].target_indices[0]
        return [
            gates.S(q0),
            gates.H(q1),
            gates.CNOT(q1, q0),
            gates.Sdag(q0),
            gates.S(q1),
            gates.H(q0),
            gates.H(q1),
        ]


class FuseRotationTranspiler(AdjacentGateFuser):
    """CircuitTranspiler, which fuses consecutive rotation gates of the same
    kind acting on the same qubit."""

    @property
    def target_gate_count(self) -> int:
        return 2

    def is_target_sequence(self, seq: Sequence[QuantumGate]) -> bool:
        left, right = seq
        return (
            left.name in [gate_names.RX, gate_names.RY, gate_names.RZ]
            and left.name == right.name
            and left.target_indices == right.target_indices
        )

    def fuse(self, seq: Sequence[QuantumGate]) -> Sequence[QuantumGate]:
        left, right = seq
        theta = (left.params[0] + right.params[0]) % (2.0 * np.pi)
        return [
            QuantumGate(
                name=left.name, target_indices=left.target_indices, params=(theta,)
            )
        ]


class NormalizeRotationTranspiler(GateKindDecomposer):
    """Normalize the parameters of the rotation gates (RX, RY, and RZ) so that
    they are in the specified range (0 to 2PI by default).

    Args:
        cycle_range: Specify a range of width 2PI in the form of (lower limit,
        upper limit). Lower limit is inclusive and upper limit is exclusive.
    """

    def __init__(
        self,
        cycle_range: tuple[float, float] = (0.0, np.pi * 2.0),
        epsilon: float = 1.0e-9,
    ):
        if not cycle_range[1] > cycle_range[0]:  # Do not accept 0 width.
            raise ValueError("Specify (lower limit, upper limit) for cycle_range.")
        if abs(cycle_range[1] - cycle_range[0] - np.pi * 2.0) > epsilon:
            raise ValueError("The width of the cycle range must be 2PI.")
        self._lower, self._upper = cycle_range

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RX, gate_names.RY, gate_names.RZ]

    def _normalize(self, theta: float) -> float:
        return ((theta - self._lower) % (np.pi * 2.0)) + self._lower

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        theta = self._normalize(gate.params[0])
        return [
            QuantumGate(
                gate.name,
                gate.target_indices,
                gate.control_indices,
                gate.classical_indices,
                (theta,),
                gate.pauli_ids,
                gate.unitary_matrix,
            )
        ]


class RX2NamedTranspiler(GateKindDecomposer):
    """Convert RX gate to Identity or X gate if it is equivalent to Identity,
    X, SqrtX, or SqrtXdag gate."""

    def __init__(self, epsilon: float = 1.0e-9):
        self._epsilon = epsilon

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RX]

    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) < self._epsilon

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta = gate.params[0] % (2.0 * np.pi)

        if self._is_close(theta, 0.0) or self._is_close(theta, 2.0 * np.pi):
            return [gates.Identity(target)]
        elif self._is_close(theta, np.pi / 2.0):
            return [gates.SqrtX(target)]
        elif self._is_close(theta, np.pi):
            return [gates.X(target)]
        elif self._is_close(theta, np.pi * 3.0 / 2.0):
            return [gates.SqrtXdag(target)]
        else:
            return [gate]


class RY2NamedTranspiler(GateKindDecomposer):
    """Convert RY gate to Identity or Y gate if it is equivalent to Identity,
    Y, SqrtY, or SqrtYdag gate."""

    def __init__(self, epsilon: float = 1.0e-9):
        self._epsilon = epsilon

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RY]

    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) < self._epsilon

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta = gate.params[0] % (2.0 * np.pi)

        if self._is_close(theta, 0.0) or self._is_close(theta, 2.0 * np.pi):
            return [gates.Identity(target)]
        elif self._is_close(theta, np.pi / 2.0):
            return [gates.SqrtY(target)]
        elif self._is_close(theta, np.pi):
            return [gates.Y(target)]
        elif self._is_close(theta, np.pi * 3.0 / 2.0):
            return [gates.SqrtYdag(target)]
        else:
            return [gate]


class RZ2NamedTranspiler(GateKindDecomposer):
    """Convert RZ gate to Identity, Z, S, Sdag, T, or Tdag gate if it is
    equivalent to a sequence of these gates."""

    def __init__(self, epsilon: float = 1.0e-9, allow_t_tdag: bool = True):
        self._epsilon = epsilon
        self._allow_t_tdag = allow_t_tdag

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RZ]

    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) < self._epsilon

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta = gate.params[0] % (2.0 * np.pi)

        if self._is_close(theta, 0.0) or self._is_close(theta, 2.0 * np.pi):
            return [gates.Identity(target)]
        elif self._is_close(theta, np.pi / 4.0):
            return [gates.T(target)] if self._allow_t_tdag else [gate]
        elif self._is_close(theta, np.pi / 2.0):
            return [gates.S(target)]
        elif self._is_close(theta, 3.0 * np.pi / 4.0):
            return [gates.S(target), gates.T(target)] if self._allow_t_tdag else [gate]
        elif self._is_close(theta, np.pi):
            return [gates.Z(target)]
        elif self._is_close(theta, 5.0 * np.pi / 4.0):
            return [gates.Z(target), gates.T(target)] if self._allow_t_tdag else [gate]
        elif self._is_close(theta, 3.0 * np.pi / 2.0):
            return [gates.Sdag(target)]
        elif self._is_close(theta, 7.0 * np.pi / 4.0):
            return [gates.Tdag(target)] if self._allow_t_tdag else [gate]
        else:
            return [gate]


class Rotation2NamedTranspiler(SequentialTranspiler):
    """Convert rotation gates (RX, RY, and RZ) to Identity, Z, S, Sdag, T, or
    Tdag gates if it is equivalent to a sequence of these gates."""

    def __init__(self, epsilon: float = 1.0e-9):
        super().__init__(
            [
                RX2NamedTranspiler(epsilon),
                RY2NamedTranspiler(epsilon),
                RZ2NamedTranspiler(epsilon),
            ]
        )


class ZeroRotationEliminationTranspiler(GateKindDecomposer):
    """Remove rotation gates (RX, RY, and RZ) with rotation angles smaller than
    epsilon."""

    def __init__(self, epsilon: float = 1.0e-9):
        self._epsilon = epsilon

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RX, gate_names.RY, gate_names.RZ]

    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) < self._epsilon

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        theta = gate.params[0] % (2.0 * np.pi)
        if self._is_close(theta, 0.0) or self._is_close(theta, 2.0 * np.pi):
            return []
        else:
            return [gate]
