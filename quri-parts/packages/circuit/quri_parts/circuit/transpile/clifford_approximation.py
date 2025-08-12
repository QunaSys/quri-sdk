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

from .gate_kind_decomposer import (
    U1ToRZTranspiler,
    U2ToRZSqrtXTranspiler,
    U3ToRZSqrtXTranspiler,
)
from .multi_pauli_decomposer import PauliRotationDecomposeTranspiler
from .transpiler import GateDecomposer


class CliffordApproximationTranspiler(GateDecomposer):
    r"""CircuitTranspiler, which replaces the non_clifford gate into sequence of
    non-parametric Clifford gates.

    If the input gate has angles, this transpiler replaces them with the
    closest value in the set :math:`\{\pi n /2| n\in\mathbb{Z}\}`. Then
    by using the rotation gate transpilers, it is decomposed into a
    sequence of non-parametric Clifford gates.
    """

    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.name not in gate_names.CLIFFORD_GATE_NAMES

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        clif_set_x = {0: "Identity", 1: "SqrtX", 2: "X", 3: "SqrtXdag"}
        clif_set_y = {0: "Identity", 1: "SqrtY", 2: "Y", 3: "SqrtYdag"}
        clif_set_z = {0: "Identity", 1: "S", 2: "Z", 3: "Sdag"}
        clif_set = {"RX": clif_set_x, "RY": clif_set_y, "RZ": clif_set_z}

        target = gate.target_indices[0]
        if gate.name == "T":
            return [gates.S(target)]

        elif gate.name == "Tdag":
            return [gates.Sdag(target)]

        elif gate.name in {"RX", "RY", "RZ", "U1", "U2", "U3", "PauliRotation"}:
            if gate.name in {"RX", "RY", "RZ"}:
                transpiled_gates: Sequence[QuantumGate] = [gate]
            if gate.name == "U1":
                transpiled_gates = U1ToRZTranspiler().decompose(gate)
            if gate.name == "U2":
                transpiled_gates = U2ToRZSqrtXTranspiler().decompose(gate)
            if gate.name == "U3":
                transpiled_gates = U3ToRZSqrtXTranspiler().decompose(gate)
            if gate.name == "PauliRotation":
                transpiled_gates = PauliRotationDecomposeTranspiler().decompose(gate)

            appro_gates = []
            for g in transpiled_gates:
                if g.name in {"RX", "RY", "RZ"}:
                    param = g.params[0]
                    angle_int = np.round(2 * param / np.pi) % 4
                    appro_gates.append(
                        QuantumGate(clif_set[g.name][angle_int], (g.target_indices[0],))
                    )
                    continue
                appro_gates.append(g)
            return appro_gates

        else:
            raise NotImplementedError(
                f"The input gate {gate.name} is not supported on the\
                CliffordApproximationTranspiler."
            )
