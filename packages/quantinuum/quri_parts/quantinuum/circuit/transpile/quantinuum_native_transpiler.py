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

import numpy as np

import quri_parts.quantinuum.circuit.gate_names as native_gate_names
from quri_parts.circuit import (
    RZ,
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    gate_names,
)
from quri_parts.circuit.transpile import CircuitTranspilerProtocol, GateKindDecomposer
from quri_parts.quantinuum.circuit import RZZ, ZZ, U1q


class RX2U1qTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes RX gates into U1q gates.

    Ref:
        [1]: https://www.quantinuum.com/hardware/h1
            System Model H1 Product Data Sheet (P4 Native Gate Set)
    """

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RX]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        return [U1q(gate.target_indices[0], gate.params[0], 0.0)]


class RY2U1qTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes RY gates into U1q gates.

    Ref:
        [1]: https://www.quantinuum.com/hardware/h1
            System Model H1 Product Data Sheet (P4 Native Gate Set)
    """

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.RY]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        return [U1q(gate.target_indices[0], gate.params[0], np.pi / 2.0)]


class H2U1qRZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes H gates into sequences of U1q and RZ
    gates.

    Ref:
        [1]: https://www.quantinuum.com/hardware/h1
            System Model H1 Product Data Sheet
            (P5 Constructed gate examples using QASM notation)
    """

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.H]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [
            U1q(target, np.pi / 2.0, -np.pi / 2.0),
            RZ(target, np.pi),
        ]


class CNOT2U1qZZRZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes CNOT gates into sequences of U1q,
    RZ, and ZZ gates.

    Ref:
        [1]: https://www.quantinuum.com/hardware/h1
            System Model H1 Product Data Sheet
            (P5 Constructed gate examples using QASM notation)
    """

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CNOT]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            U1q(target, -np.pi / 2.0, np.pi / 2.0),
            ZZ(control, target),
            RZ(control, -np.pi / 2.0),
            U1q(target, np.pi / 2.0, np.pi),
            RZ(target, -np.pi / 2.0),
        ]


class U1qNormalizeWithRZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which converts U1q gates into gate sequences
    containing RZ gates so that theta of the U1q gates will be pi or pi/2.

    Ref:
        [1]: https://www.quantinuum.com/hardware/h1
            System Model H1 Product Data Sheet (P5)
    """

    def __init__(self, epsilon: float = 1.0e-9):
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [native_gate_names.U1q]

    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) < self.epsilon

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        theta, phi = gate.params
        if self._is_close(theta, 0.0):
            return [RZ(target, 0.0)]
        elif self._is_close(theta, -np.pi / 2.0):
            return [U1q(target, -theta, phi + np.pi)]
        elif not self._is_close(theta, np.pi) and not self._is_close(
            theta, np.pi / 2.0
        ):
            return [
                U1q(target, np.pi / 2.0, phi + np.pi / 2.0),
                RZ(target, theta),
                U1q(target, np.pi / 2.0, phi - np.pi / 2.0),
            ]
        else:
            return [gate]


class CZ2RZZZTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes CZ gates into sequences of RZ and ZZ
    gates.

    Ref:
        [1]: https://www.quantinuum.com/hardware/h1
            System Model H1 Product Data Sheet (P5)
    """

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CZ]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            RZ(control, -np.pi / 2.0),
            RZ(target, -np.pi / 2.0),
            ZZ(control, target),
        ]


class CNOTRZ2RZZTranspiler(CircuitTranspilerProtocol):
    """CircuitTranspiler, which fuses consecutive CNOT and RZ gates into
    RZZ(control, target) gates if the gates are in the following sequence and
    have the following parameters.

    [CNOT(control, target), RZ(target, theta), CNOT(control, target)]

    Ref:
        [1]: https://www.quantinuum.com/hardware/h1
            System Model H1 Product Data Sheet
            (P5 Arbitrary Angle ZZ Gates)
    """

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        xs = circuit.gates
        ys = []
        i = 0

        while i < len(xs) - 2:
            a, b, c = xs[i : i + 3]  # noqa: E203
            if (
                a.name == gate_names.CNOT
                and b.name == gate_names.RZ
                and c.name == gate_names.CNOT
            ):
                control_a, target_a = a.control_indices[0], a.target_indices[0]
                target_b = b.target_indices[0]
                control_c, target_c = c.control_indices[0], c.target_indices[0]
                if (
                    control_a == control_c
                    and target_a == target_b
                    and target_b == target_c
                ):
                    ys.append(RZZ(control_a, target_a, b.params[0]))
                    i += 3
                    continue

            ys.append(xs[i])
            i += 1

        ys.extend(xs[i:])

        ret = QuantumCircuit(circuit.qubit_count)
        ret.extend(ys)
        return ret
