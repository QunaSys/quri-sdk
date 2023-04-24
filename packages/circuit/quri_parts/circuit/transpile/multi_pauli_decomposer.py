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

from quri_parts.circuit import QuantumGate, gate_names, gates

from .transpiler import GateKindDecomposer


class PauliDecomposeTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose multi-qubit Pauli gates into X, Y,
    and Z gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.Pauli]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        indices = gate.target_indices
        pauli_ids = gate.pauli_ids
        ret: list[QuantumGate] = []

        for index, pauli in zip(indices, pauli_ids):
            if pauli == 1:
                ret.append(gates.X(index))
            elif pauli == 2:
                ret.append(gates.Y(index))
            elif pauli == 3:
                ret.append(gates.Z(index))
            else:
                raise ValueError("Pauli id must be either 1, 2, or 3.")

        return ret


class PauliRotationDecomposeTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decompose multi-qubit PauliRotation gates into
    H, RX, RZ, and CNOT gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.PauliRotation]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        indices = gate.target_indices
        pauli_ids = gate.pauli_ids
        angle = gate.params[0]
        ret: list[QuantumGate] = []

        def rot_gates(rot_sign: int = 1) -> Sequence[QuantumGate]:
            rc = []
            for index, pauli in zip(indices, pauli_ids):
                if pauli == 1:
                    rc.append(gates.H(index))
                elif pauli == 2:
                    rc.append(gates.RX(index, rot_sign * np.pi / 2.0))
                elif pauli == 3:
                    pass
                else:
                    raise ValueError("Pauli id must be either 1, 2, or 3.")
            return rc

        ret.extend(rot_gates(1))
        for i in reversed(range(1, len(indices))):
            ret.append(gates.CNOT(indices[i], indices[0]))
        ret.append(gates.RZ(indices[0], angle))
        for i in range(1, len(indices)):
            ret.append(gates.CNOT(indices[i], indices[0]))
        ret.extend(rot_gates(-1))

        return ret
