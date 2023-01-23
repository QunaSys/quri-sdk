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

from quri_parts.circuit import RZ, QuantumGate, gate_names
from quri_parts.circuit.transpile import GateKindDecomposer
from quri_parts.honeywell.circuit import ZZ, U1q


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
