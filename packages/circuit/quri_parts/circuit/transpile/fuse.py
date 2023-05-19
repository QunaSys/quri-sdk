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
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    gate_names,
    gates,
)

from .transpiler import CircuitTranspilerProtocol, GateKindDecomposer


class TwoGateFuser(CircuitTranspilerProtocol, ABC):
    """Abstract base class of adjacent gates fusing transpilers."""

    @abstractmethod
    def is_target_pair(self, left: QuantumGate, right: QuantumGate) -> bool:
        ...

    @abstractmethod
    def fuse(self, left: QuantumGate, right: QuantumGate) -> Sequence[QuantumGate]:
        ...

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        xs = list(circuit.gates)
        ys: list[QuantumGate] = []

        while xs:
            x = xs.pop(0)
            if not ys:
                ys.append(x)
                continue
            y = ys.pop(-1)

            if self.is_target_pair(y, x):
                ys.extend(self.fuse(y, x))
            else:
                ys.extend([y, x])

        ret = QuantumCircuit(circuit.qubit_count)
        ret.extend(ys)
        return ret


class FuseRotationTranspiler(TwoGateFuser):
    """CircuitTranspiler, which fuses consecutive rotation gates of the same
    kind acting on the same qubit."""

    def is_target_pair(self, left: QuantumGate, right: QuantumGate) -> bool:
        return (
            left.name in [gate_names.RX, gate_names.RY, gate_names.RZ]
            and left.name == right.name
            and left.target_indices == right.target_indices
        )

    def fuse(self, left: QuantumGate, right: QuantumGate) -> Sequence[QuantumGate]:
        theta = (left.params[0] + right.params[0]) % (2.0 * np.pi)
        return [
            QuantumGate(
                name=left.name, target_indices=left.target_indices, params=(theta,)
            )
        ]


class RX2NamedTranspiler(GateKindDecomposer):
    """Convert RX gate to Identity or X gate if it is equivalent to Identity or
    X gate."""

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
        elif self._is_close(theta, np.pi):
            return [gates.X(target)]
        else:
            return [gate]


class RY2NamedTranspiler(GateKindDecomposer):
    """Convert RY gate to Identity or Y gate if it is equivalent to Identity or
    Y gate."""

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
        elif self._is_close(theta, np.pi):
            return [gates.Y(target)]
        else:
            return [gate]


class RZ2NamedTranspiler(GateKindDecomposer):
    """Convert RZ gate to Identity, Z, S, Sdag, T, or Tdag gate if it is
    equivalent to one of these gates."""

    def __init__(self, epsilon: float = 1.0e-9):
        self._epsilon = epsilon

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
        elif self._is_close(theta, np.pi):
            return [gates.Z(target)]
        elif self._is_close(theta, np.pi / 2.0):
            return [gates.S(target)]
        elif self._is_close(theta, 3.0 * np.pi / 2.0):
            return [gates.Sdag(target)]
        elif self._is_close(theta, np.pi / 4.0):
            return [gates.T(target)]
        elif self._is_close(theta, 7.0 * np.pi / 4.0):
            return [gates.Tdag(target)]
        else:
            return [gate]
