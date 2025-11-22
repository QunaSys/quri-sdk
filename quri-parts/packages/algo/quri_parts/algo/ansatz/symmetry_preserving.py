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

import numpy as np

from quri_parts.circuit import (
    CONST,
    ImmutableLinearMappedParametricQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
)

from .two_local import (
    EntanglementPatternType,
    EntLayerMakerArg,
    TwoLocal,
    build_entangler_map,
)


def _add_A_gate(
    circuit: LinearMappedParametricQuantumCircuit, arg: EntLayerMakerArg
) -> None:
    layer_index, (i, j) = arg

    theta, phi = circuit.add_parameters(
        f"theta_{layer_index}_{j}", f"phi_{layer_index}_{j}"
    )

    circuit.add_CNOT_gate(j, i)
    circuit.add_ParametricRZ_gate(j, {phi: -1.0, CONST: -np.pi})
    circuit.add_ParametricRY_gate(j, {theta: -1.0, CONST: -0.5 * np.pi})
    circuit.add_CNOT_gate(i, j)
    circuit.add_ParametricRY_gate(j, {theta: 1.0, CONST: 0.5 * np.pi})
    circuit.add_ParametricRZ_gate(j, {phi: 1.0, CONST: np.pi})
    circuit.add_CNOT_gate(j, i)


def _add_SO4_entangler(
    circuit: LinearMappedParametricQuantumCircuit, arg: EntLayerMakerArg
) -> None:
    layer_index, (i, j) = arg

    theta = circuit.add_parameter(f"theta_{layer_index}_{i}")

    circuit.add_CNOT_gate(i, j)
    circuit.add_ParametricRY_gate(i, theta)
    circuit.add_CNOT_gate(j, i)
    circuit.add_ParametricRY_gate(i, {theta: -1.0})
    circuit.add_CNOT_gate(i, j)


class SymmetryPreserving(ImmutableLinearMappedParametricQuantumCircuit):
    """
    Note:
        Simple Nakanishi-Fujii-Todo method of period 2pi does not work for
        this ansatz.

    Ref:
        Efficient Symmetry-Peserving State Preparation Circuits for the
        Variational Quantum Eigensolver Algorithm Bryan T. Gard, Linghua
        Zhu, George S. Barron, Nicholas J. Mayhall, Sophia E. Economou,
        Edwin Barnes, npj Quantum Inf 6, 10 (2020).

    Args:
        qubit_count: number of qubits.
        reps: number of layers.
        entangler_map_seq: indices specifying on which qubits each entanglement
            layer acts.

    Raises:
        ValueError: if number of qubits is less than 2.
    """

    _add_entanglement_gate = _add_A_gate

    def __init__(
        self,
        qubit_count: int,
        reps: int,
        entangler_map_seq: Optional[Sequence[Sequence[tuple[int, int]]]] = None,
    ):
        if qubit_count < 2:
            raise ValueError("number of qubit must be larger than 1")
        if entangler_map_seq is None:
            entangler_map_seq = build_entangler_map(
                qubit_count, [EntanglementPatternType.LINEAR] * reps
            )

        circuit = TwoLocal(
            qubit_count=qubit_count,
            layer_pattern="e" * reps,
            rot_layer_maker=lambda c, a: None,
            ent_layer_maker=self.__class__._add_entanglement_gate,
            rotation_indices=[],
            entangler_map_seq=entangler_map_seq,
        )

        super().__init__(circuit.get_mutable_copy())

        self.reps = reps


class SymmetryPreservingReal(SymmetryPreserving):
    """This is an alternative of Symmetry Preserving Ansatz, whose RZ rotation
    is omitted.

    Ref:
        Yohei Ibe, Yuya O. Nakagawa, Nathan Earnest, Takahiro Yamamoto, Kosuke Mitarai,
        Qi Gao, and Takao Kobayashi, Calculating transition amplitudes by variational
        quantum deflation, Phys. Rev. Research 4, 013173 (2022).

    Args:
        qubit_count: Number of qubits.
        reps: Number of layers.
        entangler_map_seq: Indices specifying on which qubits each entanglement
            layer acts.

    Raises:
        ValueError: if number of qubits is less than 2.
    """

    _add_entanglement_gate = _add_SO4_entangler
