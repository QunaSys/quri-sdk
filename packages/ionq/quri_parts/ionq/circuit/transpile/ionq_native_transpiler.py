# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from collections.abc import MutableMapping, Sequence

import numpy as np

import quri_parts.ionq.circuit.gate_names as ionq_gate_names
from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    gate_names,
    gates,
)
from quri_parts.circuit.transpile import CircuitTranspilerProtocol, GateKindDecomposer
from quri_parts.ionq.circuit import MS, XX, GPi, GPi2


class IonQNativeTranspiler(CircuitTranspilerProtocol):
    """CircuitTranspiler, which converts RX, RY, RZ, XX gates into GPi, GPi2,
    MS gates.

    The conversion process to the native gates is a direct porting of the process of
    the code in IonQ's introductory article. Due to the nature of the conversion
    process, each qubit may acquire an arbitrary extra phase. Therefore, only the
    measurement in the computational basis is guaranteed to match between before and
    after conversion.

    Ref:
        [1]: https://ionq.com/docs/getting-started-with-native-gates
    """

    def __init__(self, epsilon: float = 1.0e-6):
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def _is_close(self, x: float, y: float) -> bool:
        return abs(x - y) < self.epsilon

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:

        phase: MutableMapping[int, float] = defaultdict(float)
        cg = []

        for gate in circuit.gates:

            if gate.name == gate_names.RZ:
                target = gate.target_indices[0]
                theta = gate.params[0]
                phase[target] = (phase[target] - theta / (2.0 * np.pi)) % 1.0

            elif gate.name == gate_names.RY:
                target = gate.target_indices[0]
                theta = gate.params[0]

                if self._is_close(theta, 0.5 * np.pi):
                    cg.append(GPi2(target, (phase[target] + 0.25) % 1.0))
                elif self._is_close(theta, -0.5 * np.pi):
                    cg.append(GPi2(target, (phase[target] + 0.75) % 1.0))
                elif self._is_close(theta, np.pi):
                    cg.append(GPi(target, (phase[target] + 0.25) % 1.0))
                elif self._is_close(theta, -np.pi):
                    cg.append(GPi(target, (phase[target] + 0.75) % 1.0))
                else:
                    cg.append(GPi2(target, phase[target] % 1.0))
                    phase[target] = (phase[target] - theta / (2.0 * np.pi)) % 1.0
                    cg.append(GPi2(target, (phase[target] + 0.5) % 1.0))

            elif gate.name == gate_names.RX:
                target = gate.target_indices[0]
                theta = gate.params[0]

                if self._is_close(theta, 0.5 * np.pi):
                    cg.append(GPi2(target, phase[target] % 1.0))
                elif self._is_close(theta, -0.5 * np.pi):
                    cg.append(GPi2(target, (phase[target] + 0.5) % 1.0))
                elif self._is_close(theta, np.pi):
                    cg.append(GPi(target, phase[target] % 1.0))
                elif self._is_close(theta, -np.pi):
                    cg.append(GPi(target, (phase[target] + 0.5) % 1.0))
                else:
                    cg.append(GPi2(target, (phase[target] + 0.75) % 1.0))
                    phase[target] = (phase[target] - theta / (2.0 * np.pi)) % 1.0
                    cg.append(GPi2(target, (phase[target] + 0.25) % 1.0))

            elif gate.name == ionq_gate_names.XX:
                target0, target1 = gate.target_indices

                if gate.params[0] > 0.0:
                    cg.append(MS(target0, target1, phase[target0], phase[target1]))
                else:
                    cg.append(
                        MS(
                            target0,
                            target1,
                            phase[target0],
                            (phase[target1] + 0.5) % 1.0,
                        )
                    )

        cc = QuantumCircuit(circuit.qubit_count)
        cc.extend(cg)
        return cc


class CNOT2RXRYXXTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes CNOT gates into sequences of RX, RY,
    and XX gates.

    Ref:
        [1]: Dmitri Maslov,
            Basic circuit compilation techniques for an ion-trap quantum machine,
            New J. Phys. 19, 023035 (2017).
        [2]: https://ionq.com/docs/getting-started-with-native-gates
    """

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.CNOT]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        control, target = gate.control_indices[0], gate.target_indices[0]
        return [
            gates.RY(control, np.pi / 2.0),
            XX(control, target, np.pi / 4.0),
            gates.RX(control, -np.pi / 2.0),
            gates.RX(target, -np.pi / 2.0),
            gates.RY(control, -np.pi / 2.0),
        ]
