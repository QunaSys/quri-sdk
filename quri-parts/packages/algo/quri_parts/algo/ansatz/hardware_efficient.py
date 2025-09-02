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
from typing import Optional

from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
)

from .two_local import (
    EntanglementPatternType,
    EntLayerMakerArg,
    RotLayerMakerArg,
    TwoLocal,
    build_entangler_map,
)


def _add_ry_rz_gates(
    circuit: LinearMappedParametricQuantumCircuit, arg: RotLayerMakerArg
) -> None:
    layer_index, qubit_index = arg
    theta_y, theta_z = circuit.add_parameters(
        f"theta_{layer_index}_{qubit_index}_y", f"theta_{layer_index}_{qubit_index}_z"
    )
    circuit.add_ParametricRY_gate(qubit_index, theta_y)
    circuit.add_ParametricRZ_gate(qubit_index, theta_z)


def _add_cz_gate(
    circuit: LinearMappedParametricQuantumCircuit, arg: EntLayerMakerArg
) -> None:
    i, j = arg.qubit_indices
    circuit.add_CZ_gate(i, j)


class HardwareEfficient(ImmutableLinearMappedParametricQuantumCircuit):
    """Hardware-efficient ansatz.

    Ref:
    A. Kandala et. al.,
    Hardware-efficient variational quantum eigensolver for small molecules and quantum
    magnets,
    Nature 549, 242–246

    For application of HWE ansatz to Subspace search VQE, see:
    Nakanishi et. al.,
    Subspace-search variational quantum eigensolver for excited states,
    https://arxiv.org/abs/1810.09434

    Args:
        qubit_count: Number of qubits.
        reps: Number of repetitions of a single entanglement layer.
        rotation_indices: Qubit indices specifying on which qubits each rotation layer
          acts.
        entangler_map_seq: Qubit index pairs specifying on which qubit pairs each
          entanglement layer acts.
        sub_width: Number of qubits in a subspace for SSVQE.
        sub_reps: Number of repetitions of a single entanglement layer in a subspace
          for SSVQE.
        sub_ent_map_seq: Qubit index pairs specifying on which qubit pairs each
          entanglement layer in a subspace for SSVQE acts.
    """

    _add_rotation_gates = _add_ry_rz_gates
    _add_entanglement_gate = _add_cz_gate

    def __init__(
        self,
        qubit_count: int,
        reps: int,
        rotation_indices: Optional[Sequence[int]] = None,
        entangler_map_seq: Optional[Sequence[Sequence[tuple[int, int]]]] = None,
        sub_width: int = 0,
        sub_reps: int = 0,
        sub_ent_map_seq: Optional[Sequence[Sequence[tuple[int, int]]]] = None,
    ):
        if rotation_indices is None:
            rotation_indices = list(range(qubit_count))
        if entangler_map_seq is None:
            entangler_map_seq = build_entangler_map(
                qubit_count, [EntanglementPatternType.LINEAR] * reps
            )
        if sub_ent_map_seq is None:
            sub_ent_map_seq = build_entangler_map(
                sub_width, [EntanglementPatternType.LINEAR] * sub_reps
            )

        sub_circuit = TwoLocal(
            qubit_count,
            layer_pattern="re" * sub_reps,
            rot_layer_maker=self.__class__._add_rotation_gates,
            ent_layer_maker=self.__class__._add_entanglement_gate,
            rotation_indices=range(sub_width),
            entangler_map_seq=sub_ent_map_seq,
        )
        circuit = TwoLocal(
            qubit_count,
            layer_pattern="re" * reps + "r",
            rot_layer_maker=self.__class__._add_rotation_gates,
            ent_layer_maker=self.__class__._add_entanglement_gate,
            rotation_indices=rotation_indices,
            entangler_map_seq=entangler_map_seq,
        )

        super().__init__(sub_circuit + circuit)

        self.reps = reps


def _add_ry_gate(
    circuit: LinearMappedParametricQuantumCircuit, arg: RotLayerMakerArg
) -> None:
    layer_index, qubit_index = arg
    theta_y = circuit.add_parameter(f"theta_{layer_index}_{qubit_index}_y")
    circuit.add_ParametricRY_gate(qubit_index, theta_y)


class HardwareEfficientReal(HardwareEfficient):
    """Real-valued hardware-efficient ansatz.

    Ref:
    A. Kandala et. al.,
    Hardware-efficient variational quantum eigensolver for small molecules and quantum
    magnets,
    Nature 549, 242–246

    This implementation explicitly removes Rz gates ensuring real states.

    For application of HWE ansatz to Subspace search VQE, see:
    Nakanishi et. al.,
    Subspace-search variational quantum eigensolver for excited states,
    https://arxiv.org/abs/1810.09434

    Args:
        qubit_count: Number of qubits.
        reps: Number of repetitions of a single entanglement layer.
        rotation_indices: Qubit indices specifying on which qubits each rotation layer
          acts.
        entangler_map_seq: Qubit index pairs specifying on which qubit pairs each
          entanglement layer acts.
        sub_width: Number of qubits in a subspace for SSVQE.
        sub_reps: Number of repetitions of a single entanglement layer in a subspace
          for SSVQE.
        sub_ent_map_seq: Qubit index pairs specifying on which qubit pairs each
          entanglement layer in a subspace for SSVQE acts.
    """

    _add_rotation_gates = _add_ry_gate
