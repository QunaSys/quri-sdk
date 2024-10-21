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

from quri_parts.circuit import (
    LinearMappedParametricQuantumCircuit,
    LinearParameterMapping,
    ParameterOrLinearFunction,
    ParametricQuantumCircuitProtocol,
    ParametricQuantumGate,
    QuantumGate,
    gate_names,
    gates,
)

from .transpiler import GateKindDecomposer, ParametricCircuitTranspilerProtocol


class PauliDecomposeTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes multi-qubit Pauli gates into X, Y,
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


def rot_gates(
    rot_sign: int, indices: Sequence[int], pauli_ids: Sequence[int]
) -> Sequence[QuantumGate]:
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


class PauliRotationDecomposeTranspiler(GateKindDecomposer):
    """CircuitTranspiler, which decomposes multi-qubit PauliRotation gates into
    H, RX, RZ, and CNOT gates."""

    @property
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.PauliRotation]

    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        indices = gate.target_indices
        pauli_ids = gate.pauli_ids
        angle = gate.params[0]
        ret: list[QuantumGate] = []

        ret.extend(rot_gates(1, indices, pauli_ids))
        for i in reversed(range(1, len(indices))):
            ret.append(gates.CNOT(indices[i], indices[0]))
        ret.append(gates.RZ(indices[0], angle))
        for i in range(1, len(indices)):
            ret.append(gates.CNOT(indices[i], indices[0]))
        ret.extend(rot_gates(-1, indices, pauli_ids))

        return ret


class ParametricPauliRotationDecomposeTranspiler(ParametricCircuitTranspilerProtocol):
    """CircuitTranspiler, which decomposes multi-qubit ParametricPauliRotation
    gates into H, RX, RZ, CNOT, and ParametricRZ gates."""

    @staticmethod
    def add_decomposed_gates(
        gate: ParametricQuantumGate,
        circuit: LinearMappedParametricQuantumCircuit,
        param: ParameterOrLinearFunction,
    ) -> None:
        indices = gate.target_indices
        pauli_ids = gate.pauli_ids

        circuit.extend(rot_gates(1, indices, pauli_ids))
        for i in reversed(range(1, len(indices))):
            circuit.add_gate(gates.CNOT(indices[i], indices[0]))
        circuit.add_ParametricRZ_gate(indices[0], param)
        for i in range(1, len(indices)):
            circuit.add_gate(gates.CNOT(indices[i], indices[0]))
        circuit.extend(rot_gates(-1, indices, pauli_ids))

    def __call__(
        self, circuit: ParametricQuantumCircuitProtocol
    ) -> LinearMappedParametricQuantumCircuit:
        ret = LinearMappedParametricQuantumCircuit(
            circuit.qubit_count, circuit.cbit_count
        )
        ret._param_mapping = LinearParameterMapping(circuit.param_mapping.in_params)
        pmap = circuit.param_mapping.mapping

        for gate, param in circuit.primitive_circuit().gates_and_params:
            if isinstance(gate, QuantumGate):
                ret.add_gate(gate)
            else:
                if param is None:
                    raise ValueError("Parametric gate with no Parameter: {gate}")

                if gate.name == gate_names.ParametricRX:
                    ret.add_ParametricRX_gate(gate.target_indices[0], pmap[param])
                elif gate.name == gate_names.ParametricRY:
                    ret.add_ParametricRY_gate(gate.target_indices[0], pmap[param])
                elif gate.name == gate_names.ParametricRZ:
                    ret.add_ParametricRZ_gate(gate.target_indices[0], pmap[param])
                elif gate.name == gate_names.ParametricPauliRotation:
                    self.add_decomposed_gates(gate, ret, pmap[param])
                else:
                    raise ValueError(f"Unsupported parametric gate: {gate.name}")

        return ret
