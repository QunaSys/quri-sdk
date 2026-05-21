# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, cast

from quri_parts.circuit import ImmutableQuantumCircuit, QuantumCircuit, gates

_ECR_GATE_NAME = "ECR"


def transpile_ecr_gates(circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
    """Replace ECR gates with native gates understood by Qulacs."""
    converted = QuantumCircuit(circuit.qubit_count)
    for gate in circuit.gates:
        ecr_targets = _get_ecr_targets(gate)
        if ecr_targets is not None:
            control, target = ecr_targets
            converted.add_gate(gates.S(control))
            converted.add_gate(gates.SqrtX(target))
            converted.add_gate(gates.CNOT(control, target))
            converted.add_gate(gates.X(control))
        else:
            converted.add_gate(gate)
    return converted


def _get_ecr_targets(gate: object) -> Optional[tuple[int, int]]:
    if not hasattr(gate, "target_indices"):
        return None

    target_indices = getattr(gate, "target_indices")
    if len(target_indices) != 2:
        return None

    if hasattr(gate, "name") and gate.name == _ECR_GATE_NAME:
        return cast(tuple[int, int], tuple(target_indices))

    return None


__all__ = [
    "transpile_ecr_gates",
]
